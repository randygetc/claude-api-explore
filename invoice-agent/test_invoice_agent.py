"""Offline test harness for invoice-agent.py. No API calls, no network.

Goal: prove that when the agent says "submit" but the data is bad,
commit_approval (trusted code) still refuses, and refusing has no side effects.

Run:  python3 -m unittest -v test_invoice_agent
"""
import contextlib
import copy
import importlib.util
import io
import os
import unittest
from pathlib import Path
from unittest import mock

# The module builds an Anthropic() client at import time, which needs a key.
# A dummy is fine: nothing here ever calls the API.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

_path = Path(__file__).with_name("invoice-agent.py")  # hyphen: can't `import`
_spec = importlib.util.spec_from_file_location("invoice_agent", _path)
ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ia)

BASE_INVOICE = {
    "invoice_number": "INV-1",
    "vendor_id": "V001",
    "vendor": "Acme Corp",
    "department": "IT",
    "total": "250.00",
    "currency": "USD",
    "status": "unpaid",
}
BASE_VENDORS = {
    "V001": {"name": "Acme Corp", "status": "active"},
    "V999": {"name": "Shady LLC", "status": "suspended"},
}
BASE_BUDGETS = {"IT": {"available": "10000.00", "currency": "USD"}}


def quiet(fn, *args, **kwargs):
    """Call fn with commit_approval's debug prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class WorldTestCase(unittest.TestCase):
    """Gives every test a fresh, isolated copy of the fake accounting data."""

    def setUp(self):
        self.invoices = {"INV-1": copy.deepcopy(BASE_INVOICE)}
        self.vendors = copy.deepcopy(BASE_VENDORS)
        self.budgets = copy.deepcopy(BASE_BUDGETS)
        self.duplicates = set()
        self.approvals = {}
        self.audit_log = []
        for name, value in [
            ("INVOICES", self.invoices),
            ("VENDORS", self.vendors),
            ("BUDGETS", self.budgets),
            ("DUPLICATES", self.duplicates),
            ("APPROVALS", self.approvals),
            ("AUDIT_LOG", self.audit_log),
        ]:
            patcher = mock.patch.object(ia, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    # helpers --------------------------------------------------------------
    def commit(self, invoice_number="INV-1", note="agent says ok"):
        return quiet(ia.commit_approval, invoice_number, note)

    def assertRejected(self, result, error):
        self.assertEqual(result, {"error": error})
        self.assertEqual(self.approvals, {}, "rejection must not write an approval")
        self.assertEqual(self.audit_log, [], "rejection must not write the audit log")

    def agent_says_submit(self, before_return=None):
        """Patch run_agent to always recommend 'submit' (a worst-case agent).
        before_return runs first, to simulate the world changing mid-run."""
        def fake_run_agent(_request):
            if before_return:
                before_return()
            return ia.Recommendation(recommend="submit", reason="all checks pass")
        return mock.patch.object(ia, "run_agent", side_effect=fake_run_agent)


# 1. Happy paths -------------------------------------------------------------
class TestHappyPath(WorldTestCase):
    def test_small_invoice_gets_standard_approval(self):
        r = self.commit()
        self.assertEqual(r["status"], "pending_standard_approval")
        self.assertIs(r["payment_released"], False)

    def test_invoice_over_threshold_needs_human(self):
        self.invoices["INV-1"]["total"] = "1250.00"
        self.assertEqual(self.commit()["status"], "pending_human_approval")

    def test_exactly_at_threshold_is_standard(self):
        self.invoices["INV-1"]["total"] = "1000.00"  # policy is "over", not "at"
        self.assertEqual(self.commit()["status"], "pending_standard_approval")

    def test_budget_exactly_equal_to_total_is_allowed(self):
        self.invoices["INV-1"]["total"] = "10000.00"
        self.assertNotIn("error", self.commit())

    def test_success_writes_approval_and_audit_log(self):
        r = self.commit()
        self.assertEqual(self.approvals["invoice-approval:INV-1"], r)
        self.assertEqual(self.audit_log[0]["event"], "approval_created")
        self.assertEqual(self.audit_log[0]["approval_id"], r["approval_id"])

    def test_agent_note_is_truncated_and_labeled_untrusted(self):
        r = self.commit(note="x" * 5000)
        self.assertEqual(len(r["agent_note"]), ia.MAX_NOTE_LENGTH)
        self.assertNotIn("reason", r)


# 2. Agent approves, commit rejects: bad data the agent can miss ---------------
class TestAgentApprovesButCommitRejects(WorldTestCase):
    """Each test runs the full process_invoice() with a worst-case agent that
    always says 'submit', and checks the commit still stops it."""

    def run_flow(self, before_return=None):
        with self.agent_says_submit(before_return):
            return quiet(ia.process_invoice, "INV-1")

    def test_over_budget(self):
        self.invoices["INV-1"]["total"] = "11250.00"  # budget is 10000.00
        self.assertRejected(self.run_flow(), "Insufficient budget")

    def test_currency_mismatch(self):
        self.invoices["INV-1"]["currency"] = "EUR"
        self.assertRejected(self.run_flow(), "Budget unavailable or currency mismatch")

    def test_invoice_already_paid(self):
        self.invoices["INV-1"]["status"] = "paid"
        self.assertRejected(self.run_flow(), "Invoice is not unpaid")

    def test_invoice_pending_status(self):
        self.invoices["INV-1"]["status"] = "pending"
        self.assertRejected(self.run_flow(), "Invoice is not unpaid")

    def test_vendor_suspended(self):
        self.vendors["V001"]["status"] = "suspended"
        self.assertRejected(self.run_flow(), "Vendor not active")

    def test_vendor_missing(self):
        del self.vendors["V001"]
        self.assertRejected(self.run_flow(), "Vendor not active")

    def test_duplicate_invoice(self):
        self.duplicates.add("INV-1")
        self.assertRejected(self.run_flow(), "Duplicate invoice")

    def test_department_has_no_budget(self):
        self.invoices["INV-1"]["department"] = "Finance"
        self.assertRejected(self.run_flow(), "Budget unavailable or currency mismatch")

    def test_unknown_invoice(self):
        with self.agent_says_submit():
            r = quiet(ia.process_invoice, "INV-NOPE")
        self.assertRejected(r, "Invoice not found")


# 3. Agent checked the wrong thing: tools trust model-chosen arguments ---------
class TestAgentCheckedWrongThing(WorldTestCase):
    """The read tools take whatever key the model passes. These show the agent
    getting a clean result for the wrong record while the commit uses the
    invoice's own keys and rejects."""

    def test_wrong_vendor_looks_fine_but_invoice_vendor_is_suspended(self):
        self.invoices["INV-1"]["vendor_id"] = "V999"
        self.assertEqual(ia.get_vendor("V001")["status"], "active")  # agent's view
        self.assertRejected(self.commit(), "Vendor not active")

    def test_wrong_department_looks_fine_but_invoice_department_has_no_budget(self):
        self.invoices["INV-1"]["department"] = "Finance"
        self.assertIn("available", ia.check_budget("IT"))  # agent's view
        self.assertRejected(self.commit(), "Budget unavailable or currency mismatch")

    def test_duplicate_check_on_mistyped_number_says_clean(self):
        self.duplicates.add("INV-1")
        self.assertFalse(ia.check_duplicate("INV-l")["is_duplicate"])  # typo
        self.assertRejected(self.commit(), "Duplicate invoice")


# 4. Time-of-check / time-of-use -----------------------------------------------
class TestWorldChangesDuringAgentRun(WorldTestCase):
    def run_flow(self, mutate):
        with self.agent_says_submit(before_return=mutate):
            return quiet(ia.process_invoice, "INV-1")

    def test_invoice_paid_after_agent_looked(self):
        r = self.run_flow(lambda: self.invoices["INV-1"].update(status="paid"))
        self.assertRejected(r, "Invoice is not unpaid")

    def test_flagged_duplicate_after_agent_looked(self):
        r = self.run_flow(lambda: self.duplicates.add("INV-1"))
        self.assertRejected(r, "Duplicate invoice")

    def test_budget_spent_after_agent_looked(self):
        r = self.run_flow(lambda: self.budgets["IT"].update(available="10.00"))
        self.assertRejected(r, "Insufficient budget")


# 5. process_invoice wiring ------------------------------------------------------
class TestProcessInvoiceWiring(WorldTestCase):
    def test_reject_recommendation_never_reaches_commit(self):
        rec = ia.Recommendation(recommend="reject", reason="looks off")
        with mock.patch.object(ia, "run_agent", return_value=rec), \
             mock.patch.object(ia, "commit_approval") as commit:
            r = quiet(ia.process_invoice, "INV-1")
        commit.assert_not_called()
        self.assertEqual(r["status"], "not_submitted")
        self.assertEqual(self.approvals, {})

    def test_commit_uses_callers_invoice_number_not_models(self):
        rec = ia.Recommendation(recommend="submit", reason="INV-999 is great")
        with mock.patch.object(ia, "run_agent", return_value=rec), \
             mock.patch.object(ia, "commit_approval", return_value={}) as commit:
            quiet(ia.process_invoice, "INV-1")
        self.assertEqual(commit.call_args.args[0], "INV-1")

    def test_recommendation_schema_rejects_extra_fields(self):
        with self.assertRaises(Exception):
            ia.Recommendation(recommend="submit", reason="x", invoice_number="INV-9")

    def test_recommendation_schema_rejects_unknown_decision(self):
        with self.assertRaises(Exception):
            ia.Recommendation(recommend="approve", reason="x")


# 6. Option A guarantee: the model has no write path -----------------------------
class TestAgentIsReadOnly(unittest.TestCase):
    def test_no_write_function_is_exposed_as_a_tool(self):
        exposed = {t["name"] for t in ia.TOOLS}
        self.assertEqual(exposed, set(ia.FUNCTIONS))
        self.assertEqual(exposed, set(ia.VALIDATORS))
        for forbidden in ("commit_approval", "submit_for_approval"):
            self.assertNotIn(forbidden, exposed)

    def test_every_exposed_tool_is_strict(self):
        self.assertTrue(all(t["strict"] for t in ia.TOOLS))

    def test_execute_tool_refuses_unknown_tool(self):
        with self.assertRaises(ValueError):
            quiet(ia.execute_tool, "commit_approval",
                  {"invoice_number": "INV-1", "agent_note": "x"})

    def test_execute_tool_rejects_extra_arguments(self):
        with self.assertRaises(Exception):
            quiet(ia.execute_tool, "get_invoice",
                  {"invoice_number": "INV-9981", "total": "1.00"})


# 7. Known gaps ---------------------------------------------------------------------
# These assert the behavior we WANT. They fail today, so they are marked
# expectedFailure. When you fix the code, they flip to "unexpected success" --
# delete the decorator then.
class TestKnownGaps(WorldTestCase):
    @unittest.expectedFailure
    def test_malformed_total_returns_error_instead_of_crashing(self):
        self.invoices["INV-1"]["total"] = "$1,250.00"  # Decimal() raises today
        r = self.commit()
        self.assertIn("error", r)

    @unittest.expectedFailure
    def test_repeat_commit_rechecks_after_invoice_gets_paid(self):
        self.commit()                                  # first approval is fine
        self.invoices["INV-1"]["status"] = "paid"      # then it gets paid
        r = self.commit()                              # idempotent path today
        self.assertEqual(r, {"error": "Invoice is not unpaid"})

    @unittest.expectedFailure
    def test_approval_consumes_budget(self):
        self.invoices["INV-1"]["total"] = "6000.00"
        self.commit()
        self.invoices["INV-2"] = {**BASE_INVOICE, "invoice_number": "INV-2",
                                  "total": "6000.00"}
        r = self.commit("INV-2")                       # 12,000 total vs 10,000
        self.assertEqual(r, {"error": "Insufficient budget"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
