import unittest

from rpa_tts.orthography import CONSONANTS, ONSETS, TONES, VOWELS, expected_syllables, parse_syllable


class OrthographyTests(unittest.TestCase):
    def test_inventory_has_no_double_counted_vowel_only_syllables(self):
        inventory = expected_syllables()
        self.assertEqual((len(CONSONANTS), len(ONSETS), len(VOWELS), len(TONES)), (56, 57, 13, 7))
        self.assertEqual(len(inventory), 5187)
        self.assertEqual(sum(s.consonant == "" for s in inventory.values()), 91)
        self.assertEqual(sum(s.consonant == "" and s.tone == "_" for s in inventory.values()), 13)
        for text, syllable in inventory.items():
            self.assertEqual(parse_syllable(text), syllable)

    def test_multiletter_onsets_vowels_and_unmarked_tone(self):
        cases = {
            "Hmoob": ("hm", "oo", "b"),
            " Kuv ": ("k", "u", "v"),
            "ntxhais": ("ntxh", "ai", "s"),
            "ai": ("", "ai", "_"),
            "wm": ("", "w", "m"),
            "ca": ("c", "a", "_"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = parse_syllable(text)
                self.assertEqual((result.consonant, result.vowel, result.tone), expected)

    def test_unknown_inputs_are_not_silently_normalized(self):
        for text in ("", "hello", "kuv yog", "ca_", "ca!", "cá", "cad", "../ca"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_syllable(text)


if __name__ == "__main__":
    unittest.main()
