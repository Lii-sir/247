import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

import tools.data.make_anomaly_dataset as dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for i in range(20):
            Image.new('RGB', (64, 48), (140 + i, 170, 190)).save(self.source / f'{i}.png')
        self.args = dataset.parse_args([
            '--input-dir', str(self.source), '--output-dir', str(self.root / 'output'),
            '--train-ratio', '0.6', '--test-good-ratio', '0.2', '--test-ng-ratio', '0.2'])

    def generate(self):
        return dataset.generate_dataset(self.args, dataset.validate_args(self.args))

    def test_disjoint_counts_and_actual_defects(self):
        manifest = self.generate()
        self.assertEqual(manifest['counts'], {'train/good': 12, 'test/good': 4, 'test/ng': 4})
        self.assertEqual(len({r['source'] for r in manifest['items']}), 20)
        self.assertEqual(len({Path(r['output']).name for r in manifest['items']}), 20)
        for row in manifest['items']:
            source = self.source / row['source']
            target = self.args.output_dir / row['output']
            if row['split'] != 'test/ng':
                self.assertEqual(source.read_bytes(), target.read_bytes())
            else:
                self.assertIsNotNone(ImageChops.difference(dataset.load_rgb(source), dataset.load_rgb(target)).getbbox())
        self.assertEqual(len(list(self.source.iterdir())), 20)

    def test_duplicate_pixels_removed_and_seed_reproducible(self):
        with Image.open(self.source / '0.png') as image:
            image.save(self.source / 'copy.bmp')
        first = self.generate()
        self.assertEqual(len(first['duplicates_skipped']), 1)
        before = {r['output']: (self.args.output_dir / r['output']).read_bytes() for r in first['items']}
        self.args.output_dir = self.root / 'second'
        second = self.generate()
        self.assertEqual(first['items'], second['items'])
        for path, content in before.items():
            self.assertEqual(content, (self.args.output_dir / path).read_bytes())

    def test_overwrite_owned_only(self):
        self.generate()
        with self.assertRaises(FileExistsError):
            self.generate()
        self.args.overwrite = True
        self.generate()
        extra = self.args.output_dir / 'train/good/do_not_delete.txt'
        extra.write_text('keep')
        with self.assertRaises(ValueError):
            self.generate()
        self.assertEqual(extra.read_text(), 'keep')

    def test_unowned_folder_not_deleted(self):
        target = self.args.output_dir / 'train/good'
        target.mkdir(parents=True)
        (target / 'keep.txt').write_text('keep')
        self.args.overwrite = True
        with self.assertRaises(ValueError):
            self.generate()
        self.assertTrue((target / 'keep.txt').exists())

    def test_bad_parameters(self):
        for value in (float('nan'), float('inf'), -1, 0.8):
            self.args.train_ratio = value
            with self.assertRaises(ValueError):
                dataset.validate_args(self.args)

    def test_paths_and_corruption(self):
        for output in (self.source, self.source / 'output', self.root):
            self.args.output_dir = output
            with self.assertRaises(ValueError):
                dataset.validate_args(self.args)
        self.args.output_dir = self.root / 'fresh'
        (self.source / 'broken.png').write_text('bad')
        with self.assertRaises(OSError):
            self.generate()
        self.assertFalse(self.args.output_dir.exists())

    def test_both_defects_tiny_images_and_external_texture(self):
        for size in ((2, 2), (64, 48)):
            original = Image.new('RGB', size, (240, 240, 240))
            for kind in ('occlusion', 'dirt'):
                result, applied = dataset.make_ng_image(original, random.Random(42), (kind,), 1, 2, .04, .22, .82, [])
                self.assertEqual(result.size, size)
                self.assertIn(kind, applied)
                self.assertIsNotNone(ImageChops.difference(original, result).getbbox())
        texture = self.root / 'dirt.png'
        Image.new('RGBA', (12, 12), (20, 20, 20, 180)).save(texture)
        result = dataset.apply_dirt(original, random.Random(42), .1, .2, [texture])
        self.assertIsNotNone(ImageChops.difference(original, result).getbbox())

    def test_rounding_all_images_allocated(self):
        for n in range(1, 101):
            for ratios in ((.7, .15, .15), (0, 0, 1), (.999, .001, 0)):
                counts = dataset.allocate_counts(n, ratios)
                self.assertEqual(sum(vars(counts).values()), n)


if __name__ == '__main__':
    unittest.main()
