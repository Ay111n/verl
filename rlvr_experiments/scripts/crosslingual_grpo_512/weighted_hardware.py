"""Select dedicated sampling cards and a memory-bounded shared update card."""
import subprocess
import time


def inventory():
    output = subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],text=True)
    return {idx:(total,used,free,util) for idx,total,used,free,util in
            (map(int,line.split(",")) for line in output.splitlines())}


def sampling_cards(cfg):
    while True:
        first = inventory()
        time.sleep(3)
        second = inventory()
        cards = [g for g in cfg["sampling_gpu_pool"] if all(m[g][1]<128 and m[g][3]<5 for m in (first,second))]
        count = cfg.get("sampling_replicas", 3)
        if len(cards)>=count:
            return cards[:count]
        time.sleep(10)


def training_cards(cfg):
    while True:
        lanes = sampling_cards(cfg)
        cards = inventory()
        choices = [g for g in cfg["shared_actor_gpus"] if cards[g][2]>=cfg["actor_min_free_mib"]]
        if choices:
            actor = max(choices,key=lambda g:(cards[g][1]>=128,cards[g][2]))
            return [actor,*lanes]
        time.sleep(10)
