from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow as pa


# https://apxml.com/courses/advanced-pytorch/chapter-3-optimization-training-strategies/large-dataset-strategies
class FileDataloader(IterableDataset):
    def __init__(self, file_path):
        super().__init__()
        if (Path(file_path).suffix == '.parquet' or
                Path(file_path).suffix == '.parquet.gzip'):
            raise ValueError("The input file shoud be a '.parquet.gzip' or \
            '.parquet.gzip'.")
        self.file_path = file_path
        fileColumns = self._get_parquet_columns(file_path)
        self.label = fileColumns.pop()
        self.columns = fileColumns
        # self.dataScaler = dataScaler
        # self.processor_fn = processor_fn
        # Determine file size or number of lines/records if needed for sharding
        # self.num_records = self._get_num_records(file_path) # Example helper

    def _get_records_iterator(self):
        # Replace this with logic to iterate over your specific data
        for batch in pq.ParquetFile(self.file_path).iter_batches(1):
            # yield self.dataScaler.transform(batch.to_tensor().to_numpy())
            yield (batch.select(self.columns).to_tensor().to_numpy(),
                   batch.select([self.label]).to_pandas().to_numpy())

    def __iter__(self):
        worker_info = get_worker_info()
        record_iterator = self._get_records_iterator()
        print(record_iterator)
        if worker_info is None:  # Single-process loading
            worker_id = 0
            num_workers = 1
        else:  # Multi-process loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Basic worker sharding: each worker processes every Nth record
        # Sophisticated sharding might involve byte offsets or file splitting
        sharded_iterator = (record for i, record in enumerate(record_iterator)
                            if i % num_workers == worker_id)

        # Apply processing within the worker's iterator chain
        # processed_iterator = map(self.processor_fn, sharded_iterator)
        return sharded_iterator

    def _get_parquet_columns(self, file_path: str) -> list:
        pf = pq.ParquetFile(file_path)
        first_ten_rows = next(pf.iter_batches(batch_size=1))
        return list(pa.Table.from_batches([first_ten_rows]).to_pandas().columns)