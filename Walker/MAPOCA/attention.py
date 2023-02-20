import torch
from torch import nn


class ResidualSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size):
        """
        Модель механизма внимания, основанного на nn.MultiheadAttention ( https://arxiv.org/abs/1706.03762 )
        :param embedding_size: размер эмбедингов
        """
        super().__init__()

        # Инициализируем объект MultiheadAttention
        # Параметр num_heads подбираем экспериментально.
        # Размер эмбединга должен делиться на num_heads без остатка
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=self.num_heads)

        # полносвязный слой для вычисления query embedding
        self.fc_q = nn.Linear(embedding_size, embedding_size)

        # полносвязный слой для вычисления key embedding
        self.fc_k = nn.Linear(embedding_size, embedding_size)

        # полносвязный слой для вычисления value embedding
        self.fc_v = nn.Linear(embedding_size, embedding_size)

        # полносвязный слой для вычисления результатов работы модели
        self.fc_out = nn.Linear(embedding_size, embedding_size)

        # слой для нормализции эмбедингов
        self.embedding_norm = nn.LayerNorm(normalized_shape=embedding_size, elementwise_affine=False)

        # слой для нормализции результатов сложения выхода механизма внимания и входных эмбедингов
        self.residual_norm = nn.LayerNorm(normalized_shape=embedding_size, elementwise_affine=False)

    def forward(self, inp, mask):
        """
        Функция прямого прохождения модели внимания
        :param inp: входные эмбединги
        :return: результат работы модели
        """

        # выполним нормализацию эмбеддингов
        inp = self.embedding_norm(inp)

        # выполним перестановку N и L, потому что параметр batch_first для nn.MultiheadAttention отсутствует
        # в более ранних версиях torch, которую требует библиотека mlagents
        inp = inp.permute((1, 0, 2))

        # вычислим значения эмбеддингов для параметров query, key, value
        query = self.fc_q(inp)
        key = self.fc_k(inp)
        value = self.fc_v(inp)

        output, _ = self.attention(query=query, key=key, value=value, key_padding_mask=mask)

        # Residual
        # Складываем поэлементно результаты работы механизма внимания входные эмбеддинги и выполняем нормализацию
        output = self.fc_out(output) + inp
        output = output.permute((1, 0, 2))
        output = self.residual_norm(output)

        # Average Pooling
        numerator_mask = (~mask).unsqueeze(2)
        numerator = output * numerator_mask
        numerator = torch.sum(numerator, dim=1)
        denominator = torch.sum(~mask, dim=1, keepdim=True)
        output = numerator / denominator

        return output
