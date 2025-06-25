import json
import os
import collections
import datasets
import glob

class DetectionConfig(datasets.BuilderConfig):
    """BuilderConfig for DetectionDataset."""

    def __init__(self, base_dir, image_format=".jpg", **kwargs):
        """
        Initializes the DetectionConfig.

        Args:
            base_dir (str): The base directory where the dataset is stored.
            image_format (str): The file extension for the images (e.g., ".jpg").
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Give a name and description to this configuration
        super(DetectionConfig, self).__init__(**kwargs)
        self.base_dir = base_dir
        self.image_format = image_format

class DetectionDataset(datasets.GeneratorBasedBuilder):
    """A dataset for object detection tasks."""

    BUILDER_CONFIG_CLASS = DetectionConfig

    def _info(self):
        """Defines the dataset's features and structure."""

        annot = glob.glob(f"{self.config.base_dir}/train/*.json")[0]
        with open(annot, "r") as f:
            data = json.load(f)
            categories = [category["name"] for category in data["categories"] if category["supercategory"] and category["supercategory"]!="none"]

        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Image(),
                "width": datasets.Value("int32"),
                "height": datasets.Value("int32"),
                "objects": datasets.Sequence(
                    {
                        "id": datasets.Value("int64"),
                        "area": datasets.Value("int64"),
                        "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                        "category": datasets.ClassLabel(names=categories),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description="An object detection dataset in COCO format.",
            features=features,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators for the train, validation, and test splits."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_path": glob.glob(os.path.join(self.config.base_dir, "train", "*.json"))[0],
                    "files": glob.glob(os.path.join(self.config.base_dir, "train", f"*{self.config.image_format}")),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_path": glob.glob(os.path.join(self.config.base_dir, "validation", "*.json"))[0],
                    "files": glob.glob(os.path.join(self.config.base_dir, "validation", f"*{self.config.image_format}")),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_path": glob.glob(os.path.join(self.config.base_dir, "test", "*.json"))[0],
                    "files": glob.glob(os.path.join(self.config.base_dir, "test", f"*{self.config.image_format}")),
                },
            ),
        ]

    def _generate_examples(self, annotation_path, files):
        """Yields examples from the dataset."""

        def process_annot(annot, category_id_to_name):
            """Helper function to format a single annotation."""
            return {
                "id": annot["id"],
                "area": annot["area"],
                "bbox": annot["bbox"],
                "category": category_id_to_name[annot["category_id"]],
            }

        with open(annotation_path, "r") as f:
            data = json.load(f)

        # Create mappings for efficient lookups
        category_id_to_name = {category["id"]: category["name"] for category in data["categories"]}
        image_id_to_annotations = collections.defaultdict(list)
        for annot in data["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        
        # Maps filename to its corresponding image metadata dictionary
        filename_to_image_info = {img["file_name"]: img for img in data["images"]}

        # The main loop to generate examples
        for idx, filepath in enumerate(files):
            filename = os.path.basename(filepath)

            # Check if the image file has corresponding annotations
            if filename in filename_to_image_info:
                image_info = filename_to_image_info[filename]
                image_id = image_info["id"]

                # Process all annotations for the current image
                objects = [
                    process_annot(annot, category_id_to_name)
                    for annot in image_id_to_annotations[image_id]
                ]

                # Open the image file in binary read mode
                with open(filepath, "rb") as image_file:
                    image_bytes = image_file.read()

                # Yield the unique key (idx) and the example data
                yield idx, {
                    "image_id": image_id,
                    "image": {"path": filepath, "bytes": image_bytes},
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "objects": objects,
                }

