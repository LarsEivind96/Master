import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset("data/first_segment.tfrecord")
print(raw_dataset)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    print(example)
    example.ParseFromString(raw_record.numpy())
    print(example)
