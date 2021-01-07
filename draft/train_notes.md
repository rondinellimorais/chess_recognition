Estamos tentando entender o motivo da rede está indo ruim...

Quando ela estava ruim, eu imaginei q seu desempenho ia ruim pois a rede não conhecia uma dataset
onde as peças estavam sozinhas. Mas mesmo após treinar a rede com um dataset onde a imagem é 64x64
e só há a peça, a rede ainda continua ruim.

Então comecei a achar q o problema estava relacionado ao corte (ROI) que é realizado isso pode corte parte
essencial da peça afetando o desempenho.

Mas realizando teste percebi q o desempenho está relacionado tbm à resolução da rede.

Aqui irei anotar o desempenho das duas redes:

| Modelos |
| yolov4_best_v1.weights | Rede treinada com imagens 612 × 816    | v1, v2

## Test #1
=====

416x416
yolov4_best_v1.weights
thresh 0.8

Em uma avaliação de imagem inteira onde havia 6 peças na imagem.

**Peças**
black-queen     | 1 | 0
white-bishop    | 2 | 2
white-king      | 1 | 0 
black-king      | 1 | 1
black-bishop    | 1 | 1

A rede não teve um desempenho ruim, porém tbm não é um desempenho satisfatório.

## Test #2
=====
416x416
yolov4_best_v2.weights
thresh 0.8

Em uma avaliação de imagem inteira onde havia 6 peças na imagem.

**Peças**
black-queen     | 1 | 1
white-bishop    | 2 | 1
white-king      | 1 | - 
black-king      | 1 | 1
black-bishop    | 1 | -

A rede teve um desempenho muito ruim, inclusive não detectou algumas peças.

## Test #3
=====
416x416
yolov4_best_v2.weights
thresh 0.6

Em uma avaliação de imagem inteira onde havia 6 peças na imagem.

**Peças**
black-queen     | 1 | 1
white-bishop    | 2 | 0
white-king      | 1 | - 
black-king      | 1 | 1
black-bishop    | 1 | -

A rede teve um desempenho pior que o anterior e ainda detectou área q não são peças
de xadrez.

## Test #5
=====
96x96
yolov4_best_v1.weights
thresh 0.8

Avaliação feita em uma imagem "cropada" no photoshop considerando toda a peça

**Peças**
black-queen     | 1 | 1

A rede teve um desempenho bom identificando corretamente a peças com 91% de confiança.

## Test #6
=====
96x96
yolov4_best_v1.weights
thresh 0.8

Avaliação feita com várias imagens "cropadas" no photoshop considerando toda a peça

**Peças**
black-queen     | 1 | 1
white-bishop    | 2 | 0
white-king      | 1 | 0
black-king      | 1 | 1
black-bishop    | 1 | -

A rede teve um desempenho ruim, mesmo mudando para várias resoluções isso não melhora.
Quando treinei a rede modifiquei a resolução de 416x416 para 64x64 mas lendo a documentação percebi q nenhuma alteração deveria ser necessária, por isso estou treinando novamente com a versão v3 da base de dados e vamos ver como se sai.

## Test #7
=====
Fiz o treino que terminou as 6h do dia 06/01/2021 com a resolução de 416x416 para o dataset v3 porém o resultado ainda não é satisfatório, a rede aceita alguns erra outros não detecta outros, utilizamos thresholds de 0.6, 0.8, 0.85 e 0.92.

A documentação do darknet tem uma sessão de como melhorar a detecção do modelo, vou tentar isso.

Vou treinar um modelo mergeando os dois dataset v2 e v3, onde v2 são as imagens com visão do tabuleiro de cima e v3 são imagens cortadas pelo algoritmo de mapeamento do tabuleiro.

## Test #8
=====

Para isso eu tenho os TODO a seguir:

1. Testar e commitar as alterações locais ✅
2. Mergear os dataset v2 e v3 ✅
3. Augmentation do dataset ✅
4. Modificar o cfg conforme a sessão [How to improve object detection](https://github.com/AlexeyAB/darknet#how-to-improve-object-detection) ✅
5. Treinar o modelo não menos do que `2000 * classes` iterations conforme a documentação pede
6. Validar modelo ✅
7. Em paralelo, devemos estudar um algoritmo que seja capaz de recortar o tabuleiro sem perder a informação da peça, se conseguirmos fazer isso devemos refazer o treino do modelo com esse novo dataset. ✅