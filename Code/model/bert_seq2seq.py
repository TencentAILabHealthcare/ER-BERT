# -*- coding: utf-8 -*-

from transformers import EncoderDecoderModel

def get_EncoderDecoder_model(logger, 
                             TransformerVariant, 
                             EpitopeBert_dir, 
                             ReceptorBert_dir,
                             epitope_tokenizer,
                             receptor_tokenizer,
                             epitope_max_len,
                             receptor_max_len,
                             resume=None):
    if resume is not None:
        logger.info(f'Loading EncoderDecoder from {resume}')
        model = EncoderDecoderModel.from_pretrained(resume)
        model.config.decoder_start_token_id = receptor_tokenizer.cls_token_id
        model.config.eos_token_id = receptor_tokenizer.sep_token_id
        model.config.pad_token_id = receptor_tokenizer.pad_token_id
        model.config.vocab_size = receptor_tokenizer.vocab_size
        model.config.max_length = receptor_max_len
        return model
    
    """Load the bert model"""
    logger.info(f'Loading EpitopeBert from {EpitopeBert_dir}')
    logger.info(f'Loading ReceptorBert from {ReceptorBert_dir}')

    if TransformerVariant == 'Epitope-Receptor':
        logger.info("Using EpitopeBert as encoder, ReceptorBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            EpitopeBert_dir, ReceptorBert_dir)
        model.config.decoder_start_token_id = receptor_tokenizer.cls_token_id
        model.config.eos_token_id = receptor_tokenizer.sep_token_id
        model.config.pad_token_id = receptor_tokenizer.pad_token_id
        model.config.vocab_size = receptor_tokenizer.vocab_size
        model.config.max_length = receptor_max_len

    else:
        logger.info("Using ReceptorBert as encoder, EpitopeBert as decoder.")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            ReceptorBert_dir, EpitopeBert_dir)
        model.config.decoder_start_token_id = epitope_tokenizer.cls_token_id
        model.config.eos_token_id = epitope_tokenizer.sep_token_id
        model.config.pad_token_id = epitope_tokenizer.pad_token_id
        model.config.vocab_size = epitope_tokenizer.vocab_size
        model.config.max_length = epitope_max_len

    return model