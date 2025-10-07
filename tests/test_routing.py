import unittest

from agents.graph import classify_intent, has_citation_shape, parse_action


class IntentTests(unittest.TestCase):
    def test_imperative_action_requests_route_to_action(self):
        for query in [
            "Pause 'Chicken Burger' until 6pm",
            "unpause the Veggie Wrap",
            "Resume Fries",
            "Update opening hours to 09:00-17:00 on Sunday",
            "Please set hours for Friday to 10-22",
            "Can you pause the Chicken Burger?",
        ]:
            with self.subTest(query=query):
                self.assertEqual(classify_intent(query), "action")

    def test_how_to_questions_that_mention_actions_route_to_knowledge(self):
        for query in [
            "How do I pause a menu item?",
            "What is the policy for updating opening hours",
            "Why would an unpause not sync?",
            "How do I re-send an invoice?",
            "",
        ]:
            with self.subTest(query=query):
                self.assertEqual(classify_intent(query), "knowledge")


class ActionParsingTests(unittest.TestCase):
    def test_quoted_multiword_items_are_kept_whole(self):
        action = parse_action("Pause 'Chicken Burger' until 6pm")
        self.assertEqual(action["type"], "pause_item")
        self.assertEqual(action["item"], "Chicken Burger")
        self.assertEqual(parse_action('unpause "Spicy Bean Wrap"')["item"], "Spicy Bean Wrap")

    def test_unquoted_items_stop_at_qualifiers_and_punctuation(self):
        self.assertEqual(parse_action("pause the Chicken Burger item until 5pm")["item"], "Chicken Burger")
        self.assertEqual(parse_action("Resume Large Fries.")["item"], "Large Fries")
        self.assertEqual(parse_action("resume Large Fries")["type"], "unpause_item")

    def test_hours_changes_keep_the_requested_details(self):
        action = parse_action("Update opening hours to 09:00-17:00 on Sunday.")
        self.assertEqual(action["type"], "update_hours")
        self.assertEqual(action["details"], "Update opening hours to 09:00-17:00 on Sunday")

    def test_unparseable_requests_stay_unknown(self):
        self.assertEqual(parse_action("pause")["type"], "unknown")
        self.assertEqual(parse_action("do something")["type"], "unknown")


class CitationShapeTests(unittest.TestCase):
    def test_detects_bracketed_file_and_page(self):
        self.assertTrue(has_citation_shape("Payments go out on Fridays [Finance_Processes_Guide.pdf p1]."))
        self.assertFalse(has_citation_shape("Payments go out on Fridays."))


if __name__ == "__main__":
    unittest.main()
