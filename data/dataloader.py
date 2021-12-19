from data.dataset import IndexedRawTextDataset, LanguagePairDataset
from torch.utils.data import DataLoader


def get_my_dataloader(raw_src_fp, raw_tgt_fp, src_dict, tgt_dict, batch_size):
	src_raw_dataset = IndexedRawTextDataset(raw_src_fp, src_dict)
	tgt_raw_dataset = IndexedRawTextDataset(raw_tgt_fp, tgt_dict)

	my_dataset = LanguagePairDataset(
		src_raw_dataset,
		src_raw_dataset.sizes,
		src_dict,
		tgt_raw_dataset,
		tgt_raw_dataset.sizes,
		tgt_dict
	)

	my_dataloader = DataLoader(
		my_dataset,
		collate_fn=my_dataset.collater,
		batch_size=batch_size
	)

	return my_dataloader
