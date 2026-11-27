import os


this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_TRITON_OPS_PATH = os.path.abspath(f"{this_dir}/../")
AITER_TRITON_CONFIGS_PATH = os.path.abspath(f"{this_dir}/../configs")


def get_sage_attention_config_path(kernel, dtype, subdir="sage_attention"):
    from .arch_info import get_arch, get_num_sms

    arch, num_cu = get_arch(), get_num_sms()
    directory = os.path.join(AITER_TRITON_CONFIGS_PATH, subdir)

    def config_path(label):
        return os.path.join(directory, f"{kernel}-device={label}-dtype={dtype}.json")

    packaged = config_path(f"{arch}_cu_{num_cu}")
    if os.path.isfile(packaged):
        return packaged

    # hcutuner uses _cu80 in newer releases and :cu_80 in older ones.
    labels = [f"{arch}_cu{num_cu}"]
    if os.name != "nt":
        labels.append(f"{arch}:cu_{num_cu}")
    for label in labels:
        path = config_path(label)
        if os.path.isfile(path):
            return path
    return packaged
