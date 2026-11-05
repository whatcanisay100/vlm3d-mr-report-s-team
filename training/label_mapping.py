from __future__ import annotations
import numpy as np
import torch



COPY_MAP = {
    "Lacunar infarct": "lacunar_stroke",
    "Arachnoid cyst": "arachnoid_cyst",
    "Cyst of pineal gland": "pineal_cyst",
    "Glioma": "glioma",
    "Metastatic malignant neoplasm to brain": "brain_metastasis",
    "Intracranial meningioma": "meningioma",
    "Pituitary adenoma": "pituitary_tumor",
    "Rathke's pouch cyst": "rathkes_cleft_cyst",
    "Chiari malformation": "chiari_malformation",
    "Cerebral atrophy": "cerebral_atrophy",
    "Cerebral edema": "cerebral_edema",
    "Structure of cave of septum pellucidum": "cavum_septum_pellucidum",
    "Mega cisterna magna": "mega_cisterna_magna",
    "Ventriculomegaly": "ventriculomegaly",
    "Cavernous hemangioma": "cavernous_malformation_cavernoma",
    "Demyelinating disease of central nervous system": "multiple_sclerosis",
    "Spinal stenosis": "spinal_degenerative_changes",
    "Cerebral infarction": "ischemic_3x",          
    "Cerebral hemorrhage": "hemorrhage_2x",        
    "Encephalomalacia": "max_with_gliosis",        
    "Gliosis": "max_with_gliosis",
}


MASKED_OFFICIAL = {
    "brain_tumor", "head_neck_tumor", "spine_tumor", "intraventricular_tumor",
    "hydrocephalus_ex_vacuo", "craniotomy_craniectomy", "tumor_resection_cavity",
    "postsurgical_changes",
}


class MLPHead(torch.nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 1024), torch.nn.BatchNorm1d(1024), torch.nn.GELU(),
            torch.nn.Linear(1024, 512), torch.nn.BatchNorm1d(512), torch.nn.GELU(),
            torch.nn.Linear(512, classes))

    def forward(self, x):
        return self.net(x)


def build_mapping(keep32: list[str], names37: list[str]):
    """返回 (37idx→[32idx列表], 32idx→(op, 37idx), masked_32idx 集合)."""
    idx37 = {n: i for i, n in enumerate(names37)}
    src = {k: (v, idx37[k]) for k, v in COPY_MAP.items()}
    col_map = {}          
    for k32 in keep32:
        i32 = keep32.index(k32)
        if k32 in MASKED_OFFICIAL:
            col_map[i32] = []
        elif k32 == "acute_ischemic_stroke":
            col_map[i32] = [("copy", src["Cerebral infarction"][1])]
        elif k32 == "chronic_ischemic_stroke":
            col_map[i32] = [("copy", src["Cerebral infarction"][1])]
        elif k32 == "small_vessel_ischemic_disease":
            col_map[i32] = [("copy", src["Cerebral infarction"][1])]
        elif k32 == "intracranial_hemorrhage":
            col_map[i32] = [("copy", src["Cerebral hemorrhage"][1])]
        elif k32 == "intraparenchymal_hemorrhage":
            col_map[i32] = [("copy", src["Cerebral hemorrhage"][1])]
        elif k32 == "encephalomalacia_gliosis":
            col_map[i32] = [("max", src["Encephalomalacia"][1]),
                            ("max", src["Gliosis"][1])]
        elif k32 == "brain_mass_effect":
            col_map[i32] = [("copy", src["Cerebral edema"][1])]
        else:
            
            hit = [k for k, v in COPY_MAP.items() if v == k32]
            col_map[i32] = [("copy", idx37[hit[0]])] if hit else []
    return col_map


def map_labels(y37: np.ndarray, keep32, names37) -> tuple[np.ndarray, np.ndarray]:
    col_map = build_mapping(keep32, names37)
    y32 = np.zeros((y37.shape[0], len(keep32)), dtype=np.float32)
    mask = np.zeros(len(keep32), dtype=bool)
    for i32, ops in col_map.items():
        if not ops:
            mask[i32] = True          
            continue
        vals = [y37[:, j] for op, j in ops]
        y32[:, i32] = np.maximum.reduce(vals) if len(vals) > 1 else vals[0]
    return y32, mask
