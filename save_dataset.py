from datasets import load_dataset
  

for split in ["train", "validation", "test"]:
	test_dataset = load_dataset("asapp/slue-phase-2", name="hvb", split=split)

	# test_dataset.save_to_disk(f"/work/yuxiang1234/backup/slue-sa/{split}.hf")
	test_dataset.save_to_disk(f"slue-dac/{split}.hf")

for split in ["train", "validation", "test"]:
	test_dataset = load_dataset("asapp/slue", name="voxceleb", split=split)
	test_dataset.save_to_disk(f"slue-sa/{split}.hf")
