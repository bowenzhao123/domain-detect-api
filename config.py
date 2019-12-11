import numpy as np

class DataConfig:
    # bucket = 'pearl-data'
    raw_data_path = 'data/raw_data.csv'
    pred_raw_data_path = 'predict/pred_raw_data.csv'
    high_confidence_path = 'predict/high_confidence.csv'
    low_confidence_path = 'predict/low_confidence.csv'
    kmean_clusters_path = 'data/kmean_clusters_parameter.pkl'
    nmf_clusters_path = 'data/nmf_clusters_parameter.pkl'
    word_freq_path = {
        'script':'data/script_word_freq.pkl',
        'img':'data/img_word_freq.pkl',
        'href':'data/href_word_freq.pkl',
        'text':'data/text_word_freq.pkl',
        'struct':'data/struct_word_freq.pkl'
    }

    bow_data_path = {
        'script':'data/script_bow_data.pkl',
        'img':'data/img_bow_data.pkl',
        'href':'data/href_bow_data.pkl',
        'text':'data/text_bow_data.pkl',
        'struct':'data/struct_bow_data.pkl'
    }
    tf_data_path = {
        'script':'data/script_tf_data.pkl',
        'img':'data/img_tf_data.pkl',
        'href':'data/href_tf_data.pkl',
        'text':'data/text_tf_data.pkl',
        'struct':'data/struct_tf_data.pkl'
    } 

    pred_bow_data_path = {
        'script':'predict/script_bow_data.pkl',
        'img':'predict/img_bow_data.pkl',
        'href':'predict/href_bow_data.pkl',
        'text':'predict/text_bow_data.pkl',
        'struct':'predict/struct_bow_data.pkl'
    }

    pred_tf_data_path = {
        'script':'predict/script_tf_data.pkl',
        'img':'predict/img_tf_data.pkl',
        'href':'predict/href_tf_data.pkl',
        'text':'predict/text_tf_data.pkl',
        'struct':'predict/struct_tf_data.pkl'
    }  
    
    kmean_data_path = {
        'script':'data/script_kmean_data.pkl',
        'img':'data/img_kmean_data.pkl',
        'href':'data/href_kmean_data.pkl',
        'text':'data/text_kmean_data.pkl',
        'struct':'data/struct_kmean_data.pkl'
    }

    nmf_data_path = {
        'script':'data/script_nmf_data.pkl',
        'img':'data/img_nmf_data.pkl',
        'href':'data/href_nmf_data.pkl',
        'text':'data/text_nmf_data.pkl',
        'struct':'data/struct_nmf_data.pkl'
    }

    trans_data = 'data/trans_data.pkl'
    pred_trans_data_path = 'predict/pred_trans_data.pkl'

    rule_train_save_path = 'data/rule_train_result.csv'
    rule_ped_save_path = 'predict/rule_pred_result.csv'
    pred_result_save_path = 'predict/pred_result.csv'


class ModelConfig:
    # bucket = 'pearl-model'
    bow_model_path = {
        'script':'model/script_bow_model.pkl',
        'img':'model/img_bow_model.pkl',
        'href':'model/href_bow_model.pkl',
        'text':'model/text_bow_model.pkl',
        'struct':'model/struct_bow_model.pkl'
    }
    tf_model_path = {
        'script':'model/script_tf_model.pkl',
        'img':'model/img_tf_model.pkl',
        'href':'model/href_tf_model.pkl',
        'text':'model/text_tf_model.pkl',
        'struct':'model/struct_tf_model.pkl'
    } 
    nmf_model_path = {
        'script':'model/script_tf_model.pkl',
        'img':'model/img_tf_model.pkl',
        'href':'model/href_tf_model.pkl',
        'text':'model/text_tf_model.pkl',
        'struct':'model/struct_tf_model.pkl'
    }  
    pred_tf_data_path = None
    provider_vec_path = 'model/provider_vec.pkl' 
    kmean_model_path = {
        'script':'model/script_kmean_model.pkl',
        'img':'model/img_kmean_model.pkl',
        'href':'model/href_kmean_model.pkl',
        'text':'model/text_kmean_model.pkl',
        'struct':'model/struct_kmean_model.pkl'
    } 
    nmf_model_path = {
        'script':'model/script_nmf_model.pkl',
        'img':'model/img_nmf_model.pkl',
        'href':'model/href_nmf_model.pkl',
        'text':'model/text_nmf_model.pkl',
        'struct':'model/struct_nmf_model.pkl'
    } 
    ml_model_path = 'model/ml_model.pkl'

class MLConfig:
    rf_model_path = 'model/ml_model.pkl'
    train_pred_save_file = 'data/train_pred.csv'
    ml_pred_save_file = 'predict/ml_pred.csv' 
    split = .2
    rf_parameters = {
        'n_estimators':np.arange(10,30,2),
        'max_depth':np.arange(4,15,2)
    }
    accuracy_res_path = 'data/acc_result.pkl'