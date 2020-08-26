from .bert_ner_train.index import train

def main(data, model, args):
  epochs = 10 if not hasattr(args, 'epochs') else args.epochs
  batchSize = 16 if not hasattr(args, 'batchSize') else args.batchSize
  modelPath = None if not hasattr(args, 'modelPath') else args.modelPath

  trainCsvPath = data.trainCsvPath

  trainParam = model.config
  trainParam.update({
    "ner": model.model,
    "data_dir": trainCsvPath,
    "output_dir": modelPath,
    "train_batch_size": batchSize,
    "num_train_epochs": epochs
  })

  train(trainParam)

  return model