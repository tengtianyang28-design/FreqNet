from detectron2.config import CfgNode as CN

def add_student_teacher_config(cfg):
    cfg.MODEL.USE_SCORE_THRESH = False
    cfg.THRESH_PATTERN = False
    cfg.FIXMATCH = False
    cfg.FIXMATCH_STRONG_AUG = False
    cfg.FIXMATCH_BBOX_ERASE_SCALE = (0.4, 0.7)
    cfg.FIXMATCH_BBOX_ERASE_SCALE_INFERENCE = (0.01, 0.05)
    cfg.FIXMATCH_BBOX_ERASE_RATIO = (0.3, 3.3)
    cfg.MASK_BOXES = 0
    cfg.MASK_BOXES_THRESH = 0.9
    cfg.MASK_BOXES_RPN = False
    cfg.DET_THRESH = 0.8
    cfg.DISTILLATION_LOSS_WEIGHT = 0.0
    cfg.CONSISTENCY_REGULARIZATION = False
    cfg.MODEL.USE_FREQUENCY_REFINEMENT = False
    cfg.MODEL.FREQUENCY_REFINEMENT_GATE = True
    cfg.MODEL.FREQUENCY_REFINEMENT_LEVELS = None
    # Dual-path refinement (Innovation Point 2)
    cfg.MODEL.USE_DUAL_PATH_REFINEMENT = False
    cfg.MODEL.DUAL_PATH_KERNEL_SIZE = 3
    cfg.MODEL.DUAL_PATH_NUM_HEADS = 4
    cfg.MODEL.DUAL_PATH_DROPOUT = 0.1
    cfg.MODEL.DUAL_PATH_REDUCTION_RATIO = 4
    cfg.MODEL.DUAL_PATH_REFINEMENT_LEVELS = None
    cfg.PRIOR_FILTER = CN()
    cfg.PRIOR_FILTER.ENABLED = False
    cfg.PRIOR_FILTER.PATH = ""
    cfg.PRIOR_FILTER.HIGH_VOTE = 2
    cfg.PRIOR_FILTER.MEDIUM_VOTE = 1
    cfg.PRIOR_FILTER.MEDIUM_SCORE = 0.9
    cfg.PRIOR_FILTER.MIN_SCORE = 0.7
    # Keep prior-related keys to satisfy legacy configs; not used by our code
    if not hasattr(cfg.MODEL, "ROI_BOX_HEAD"):
        cfg.MODEL.ROI_BOX_HEAD = CN()
    cfg.MODEL.ROI_BOX_HEAD.USE_PRIOR = False
    cfg.MODEL.ROI_BOX_HEAD.PRIOR_ALPHA = 1.0
    cfg.MODEL.ROI_BOX_HEAD.PRIOR_PATH = ""
