'''
train and save the model pipeline
'''
from setting import config
from utils.audio import create_audio_loader
from utils.berthelper import create_text_loader
from utils.multi import create_multi_loader
from utils import train


if config.role =='r':

    labels=config.R_labels
    n_train=len(config.train_r)
    n_valid=len(config.valid_r)
    
    if config.mode == "text":
        train_dataloader = create_text_loader(config.train_r, shuffle=True)
        val_dataloader = create_text_loader(config.valid_r, shuffle=False)
        test_dataloader = create_text_loader(config.test_r, shuffle=False)
        path=config.R_text_model_path

    elif config.mode == "audio":
        train_dataloader = create_audio_loader(config.train_r, config.train_img_r, shuffle=True)
        val_dataloader = create_audio_loader(config.valid_r, config.test_img_p ,shuffle=False)
        test_dataloader = create_audio_loader(config.test_r, config.test_img_p ,shuffle=False)
        path=config.R_audio_model_path

    elif config.mode == "multi":
        train_dataloader = create_multi_loader(config.train_r, config.train_img_r,shuffle=True)
        val_dataloader = create_multi_loader(config.valid_r,config.test_img_r, shuffle=False)
        test_dataloader = create_multi_loader(config.test_r, config.test_img_r,shuffle=False)

elif config.role=='p':

    labels=config.P_labels
    n_train=len(config.train_p)
    n_valid=len(config.valid_p)

    if config.mode == "text":
        train_dataloader = create_text_loader(config.train_p, shuffle=True)
        val_dataloader = create_text_loader(config.valid_p, shuffle=False)
        test_dataloader = create_text_loader(config.test_p, shuffle=False)
        path=config.P_text_model_path

    elif config.mode == "audio":
        train_dataloader = create_audio_loader(config.train_p, config.train_img_p, shuffle=True)
        val_dataloader = create_audio_loader(config.valid_p, config.test_img_p ,shuffle=False)
        test_dataloader = create_audio_loader(config.test_p, config.test_img_p ,shuffle=False)
        path=config.P_audio_model_path

    elif config.mode == "multi":
        train_dataloader = create_multi_loader(config.train_p, config.train_img_p,shuffle=True)
        val_dataloader = create_multi_loader(config.valid_p,config.test_img_p, shuffle=False)
        test_dataloader = create_multi_loader(config.test_p, config.test_img_p,shuffle=False)

# main
if __name__ == '__main__':
    train.start_train(config,train_dataloader, val_dataloader,test_dataloader, labels, n_train, n_valid, path)



