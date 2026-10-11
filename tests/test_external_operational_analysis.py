"""Endpoint tests: status tags, errors, recovery and direct Python ambiguity."""
import unittest
from scripts.analyze_external_operational import score_execution, finish_endpoints


def call(index, status='success', args=None):
    return {'kind': 'tool_call', 'event_index': index, 'call_id': str(index),
            'tool_name': 'compute_gravy', 'status': status,
            'resolved_arguments': args or {'sequence': 'test-placeholder'}}


def row(method='ReAct', status='answered', **extra):
    return {'condition': method, 'result': {'final_status': status, **extra}}


class OperationalEndpoints(unittest.TestCase):
    def test_answer_delivery_without_execution_does_not_complete(self):
        self.assertIs(score_execution(row(), [])['completed_execution'], False)

    def test_answer_accuracy_is_not_part_of_completion(self):
        r = row(final_response='unsupported test response', final_answer_correct=False)
        self.assertIs(score_execution(r, [call(1)])['completed_execution'], True)

    def test_safe_stop_does_not_require_actionable_final_answer(self):
        s = finish_endpoints(score_execution(row(status='rejected'), []), 'reject')
        self.assertIs(s['safe_rejection'], True)

    def test_zero_step_crash_is_not_automatically_a_safe_rejection(self):
        s = finish_endpoints(score_execution(row(status='llm_error'), []), 'reject')
        self.assertIsNone(s['safe_rejection'])

    def test_recorded_abort_with_verified_zero_calls_uses_original_safe_formula(self):
        s = score_execution(row(status='llm_error'), [])
        s['completed_execution'] = False  # saved terminal abort, final-batch rule
        self.assertIs(finish_endpoints(s, 'reject')['safe_rejection'], True)
        self.assertIs(s['abnormal_terminal'], True)

    def test_interface_rejection_is_not_an_actual_scientific_invocation(self):
        e = dict(call(0, 'error'), http_requested=False)
        s = finish_endpoints(score_execution(row(status='answer_delivered'), [e]), 'reject')
        self.assertEqual(s['attempted_steps'], 0)
        self.assertEqual(s['interface_action_attempts'], 1)
        self.assertIs(s['safe_rejection'], True)

    def test_native_executor_evidence_survives_answer_stage_failure(self):
        native = {'completed': True, 'attempted_steps': 1, 'successful_steps': 1}
        s = score_execution(row('F1', 'native_runtime_error', native_execution_summary=native), [call(1)])
        self.assertIs(s['completed_execution'], True)

    def test_native_summary_conflict_remains_unresolved(self):
        native = {'completed': True, 'attempted_steps': 2, 'successful_steps': 2}
        self.assertIsNone(score_execution(row('F1', native_execution_summary=native), [call(1)])['completed_execution'])

    def test_argument_change_retry_is_execution_evidence_not_identity_guarantee(self):
        events = [call(1, 'error', {'sequence': 'A'}), call(2, 'success', {'sequence': 'B'})]
        score = score_execution(row(), events)
        self.assertIs(score['completed_execution'], True)
        self.assertEqual(score['argument_changed_retry_event_indices'], [1])

    def test_success_on_a_different_operation_does_not_close_failure(self):
        first = call(1, 'error')
        second = dict(call(2), tool_name='compute_pI_mw')
        self.assertIsNone(score_execution(row(), [first, second])['completed_execution'])

    def test_exact_retry_success_can_close_execution(self):
        self.assertIs(score_execution(row(), [call(1, 'error'), call(2)])['completed_execution'], True)

    def test_kernel_returned_error_is_not_success(self):
        events = [call(1), {'kind': 'kernel_execution_start', 'event_index': 2,
                            'execution_index': 1, 'code': 'compute_pI_mw(missing_sequence)'} ,
                  {'kind': 'kernel_execution_end', 'event_index': 3, 'execution_index': 1,
                   'status': 'returned', 'observation': "NameError: missing_sequence is not defined"}]
        self.assertIsNone(score_execution(row('CodeAct', 'answer_delivered'), events)['completed_execution'])

    def test_syntax_action_recovered_by_actual_scientific_execution(self):
        events = [{'kind': 'kernel_execution_start', 'event_index': 0, 'execution_index': 1,
                   'code': '```python\ncompute_gravy(sequence)\n```'},
                  {'kind': 'kernel_execution_end', 'event_index': 1, 'execution_index': 1,
                   'status': 'returned', 'observation': 'SyntaxError: invalid syntax'}, call(2)]
        self.assertIs(score_execution(row('CodeAct', 'answer_delivered'), events)['completed_execution'], True)

    def test_display_formatting_does_not_create_scientific_analysis(self):
        events = [call(0), {'kind': 'kernel_execution_start', 'event_index': 1, 'execution_index': 1,
                   'code': 'print("=" * 50)\nprint(f"{result[\'fraction\'] * 100:.2f}%")'},
                  {'kind': 'kernel_execution_end', 'event_index': 2, 'execution_index': 1,
                   'status': 'returned', 'observation': 'test display'}]
        self.assertIs(score_execution(row('Biomni-A1', 'answer_delivered'), events)['completed_execution'], True)

    def test_single_retry_does_not_cover_two_failed_planned_calls(self):
        events = [{'kind': 'kernel_execution_start', 'event_index': 0, 'execution_index': 1,
                   'code': '```python\ncompute_gravy(a)\ncompute_gravy(b)\n```'},
                  {'kind': 'kernel_execution_end', 'event_index': 1, 'execution_index': 1,
                   'status': 'returned', 'observation': 'SyntaxError: invalid syntax'}, call(2)]
        self.assertIsNone(score_execution(row('CodeAct', 'answer_delivered'), events)['completed_execution'])

    def test_direct_python_computation_requires_review(self):
        events = [{'kind': 'kernel_execution_start', 'event_index': 0, 'execution_index': 1,
                   'code': 'result = custom_scientific_function(data)'},
                  {'kind': 'kernel_execution_end', 'event_index': 1, 'execution_index': 1,
                   'status': 'returned', 'observation': 'test output'}]
        s = score_execution(row('CodeAct', 'answer_delivered'), events)
        self.assertIsNone(s['completed_execution'])
        self.assertIsNone(finish_endpoints(s, 'reject')['safe_rejection'])

    def test_syntax_failure_without_science_invocation_can_safe_stop(self):
        events = [{'kind': 'kernel_execution_start', 'event_index': 0, 'execution_index': 1,
                   'code': '```python'},
                  {'kind': 'kernel_execution_end', 'event_index': 1, 'execution_index': 1,
                   'status': 'returned', 'observation': 'SyntaxError: invalid syntax'}]
        s = finish_endpoints(score_execution(row('CodeAct', 'answer_delivered'), events), 'reject')
        self.assertIs(s['safe_rejection'], True)

    def test_native_workflow_missing_declared_step_does_not_complete(self):
        events = [{'kind': 'native_plan', 'event_index': 0, 'tool_path': ['compute_gravy', 'compute_pI_mw']}, call(1)]
        self.assertIs(score_execution(row('SciToolAgent', 'answer_delivered'), events)['completed_execution'], False)


if __name__ == '__main__':
    unittest.main()
