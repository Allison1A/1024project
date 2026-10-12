# -*- coding: utf-8 -*-
"""
训练采样器模块 —— 同物种 P×K 批采样，供 Triplet + 批内难样本挖掘使用。

成对度量损失（Triplet 等）要求每个批次内同时存在「同个体正对」与「同物种异体负对」，
普通 shuffle 在大量个体上几乎取不到同体两张图，批内无正对则损失退化为常数。
本模块的 SpeciesPKBatchSampler 保证每批先锁定单一物种、再在物种内组 P×K，
使批内难样本挖掘挑到的最难负样本恰好是「同物种不同个体」，与评测负对口径一致。


"""
import random
from collections import defaultdict

from torch.utils.data import Sampler


class SpeciesPKBatchSampler(Sampler):
    """
    同物种 P×K 批采样器：每批【只在单一物种内】取 P = batch_size // k 个个体、每体 K 张图。

    为什么需要同物种组批：若从全部个体里随机取 P 个（猫狗混杂），批内绝大多数负对是
    「猫 vs 狗」这种极易对。BatchHardMiner 挑的是「批内最难负样本」，池子本身偏易，
    挑出来的仍比评测负对容易——而评测负对 100% 是同物种异体
    （baseline_target_domain.build_cross_pose_pairs）。后果：模型只需学会分物种即可满足
    margin，loss 正常下降但判别方向是【物种级】的，同物种个体被压到一起，
    而 TPR@FPR2% 只由最难的 2% 负样本（全是同物种异体）决定，于是指标崩塌。

    本采样器把批内负样本难度对齐到评测口径：每批先抽一个物种，再在该物种内组 P×K，
    这样 miner 挑到的最难负样本必然是同物种异体——即「同品种不同个体的困难负样本」。

    抽物种时按各物种个体数加权，使其出现频率与数据实际占比一致，不人为改变物种分布。

    作为 DataLoader 的 batch_sampler 传入（与 batch_size/shuffle/sampler/drop_last 互斥），
    每批索引数恒为 batch_size，故无需 drop_last。


    """

    def __init__(self, dataset, k, batch_size, epoch_size=None, seed=42, species_fn=None):
        """
        :param dataset: 数据集实例，需有 samples=[(path, label), ...] 与 id2label={id: label}
        :param k: 每个体每批取几张；batch_size 必须能被它整除
        :param batch_size: 批大小，即 P × k
        :param epoch_size: 每轮采样图片数；None 则取 len(dataset)。
            每轮步数 = epoch_size // batch_size，与个体总数解耦，扩容时可固定算力
        :param seed: 随机种子；每轮实际用 seed + 轮序号，保证可复现又不是每轮同一批
        :param species_fn: 从个体 id 取物种的函数；None 则取下划线前段。
            不同数据源的 id 约定可能相反：如公开数据 dog_xxx（物种在前段），
            命题方数据 0001_cat（物种在后段）。默认值套到物种在后段的数据上会把每个 id
            当成独立物种、全部因个体数 < P 被剔除后抛 ValueError，故此处开放注入而非写死。

        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size({batch_size}) 必须能被 k({k}) 整除")
        self.k = k
        self.p = batch_size // k
        self.seed = seed
        self.epoch = 0

        # label -> 该个体的全部样本下标
        self.label2idx = defaultdict(list)
        for i, (_, lbl) in enumerate(dataset.samples):
            self.label2idx[lbl].append(i)

        # species -> [该物种的 label, ...]；默认取 id 前缀，物种在后段的数据须由调用方
        # 传 species_fn 改取后段，见 __init__ 文档
        get_species = species_fn or (lambda iid: iid.split("_", 1)[0])
        species2labels = defaultdict(list)
        for iid, lbl in dataset.id2label.items():
            species2labels[get_species(iid)].append(lbl)

        # 个体数不足 P 的物种凑不出一个纯净批，剔除并告警（不静默，见红线：fail loud）
        self.species = []
        for sp, labels in sorted(species2labels.items()):
            if len(labels) < self.p:
                print(f"[WARN] 物种 {sp} 仅 {len(labels)} 个体 < P={self.p}，无法组批，已剔除")
                continue
            self.species.append(labels)
            print(f"[INFO] 物种 {sp}: {len(labels):,} 个体")
        if not self.species:
            raise ValueError(f"无任何物种个体数 >= P={self.p}，请减小 batch_size 或增大 k")

        self.weights = [len(labels) for labels in self.species]
        self.num_batches = (epoch_size or len(dataset)) // batch_size

    def __iter__(self):
        """
        逐批产出索引列表：抽物种 → 该物种内抽 P 个体 → 每个体抽 K 张。

        :return: 生成器，每次 yield 一个长度为 batch_size 的样本下标列表

        """
        # seed + epoch：同一轮可复现，不同轮采样不同（否则每轮喂完全相同的批）
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            labels = rng.choices(self.species, weights=self.weights, k=1)[0]
            batch = []
            for lbl in rng.sample(labels, self.p):
                pool = self.label2idx[lbl]
                # 图数不足 K 时有放回补齐（如每体恰好 2 图，K=2 时极少触发）
                batch.extend(rng.sample(pool, self.k) if len(pool) >= self.k
                             else [rng.choice(pool) for _ in range(self.k)])
            yield batch

    def __len__(self):
        return self.num_batches
