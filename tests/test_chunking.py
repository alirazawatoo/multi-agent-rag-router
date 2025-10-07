import unittest
from ingestion.chunking import split_text
class ChunkingTests(unittest.TestCase):
    def test_invalid_sizes_fail_instead_of_looping_forever(self):
        for size,overlap in [(0,0),(-1,0),(10,10),(10,11),(10,-1)]:
            with self.assertRaises(ValueError):split_text("synthetic document",size,overlap)
    def test_valid_chunks_preserve_the_entire_document(self):
        source="0123456789abcdefghijABCDEFGHIJ"
        chunks=split_text(source,10,2)
        rebuilt=chunks[0]+"".join(chunk[2:] for chunk in chunks[1:])
        self.assertEqual(rebuilt,source)
        self.assertTrue(all(len(chunk)<=10 for chunk in chunks))
    def test_empty_and_short_input(self):
        self.assertEqual(split_text(""),[])
        self.assertEqual(split_text("synthetic"),["synthetic"])
if __name__ == "__main__":unittest.main()
