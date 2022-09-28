from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained('roberta.large', checkpoint_file='model.pt')
roberta.eval()
