import tensorflow as tf

def init():
    global use_tpu, tpu_name, task_name, do_train, do_eval, do_predict, do_train_sample_masks, do_predict_sample_masks, test_file, train_file
    global data_dir, vocab_file, bert_config_file, init_checkpoint, max_seq_length, train_batch_size, learning_rate, get_tokenized_text
    global num_train_epochs, output_dir, apply_rp, pruning_mask, ckpt_eval_path, test_labels_known, weighted_loss, freeze_bert_weights
