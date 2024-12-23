from typing import Tuple
from VGmsdnDataset import SNPSingleData


def build_loader_SNP(cfg, SCENE_NM: str, PROMPT: str, CANVAS_SIZE: Tuple):
    data_dir = cfg['TEST']['SAVE_DIR']
    single_data = SNPSingleData(scene_nm=SCENE_NM, prompt=PROMPT, canvas_size=CANVAS_SIZE, save_dir=data_dir,
                                gen_sg=True)

    assert cfg['MODEL']['PRETRAIN'] == False
    return single_data
