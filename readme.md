# Previsão de Preço por Quantidade

Aplicação web que utiliza uma rede neural simples para prever o preço total com base na quantidade de itens informada. O modelo é treinado e executado diretamente no navegador, sem necessidade de servidor ou backend.

## Tecnologias

- HTML5
- CSS3
- JavaScript (ES6+)
- [TensorFlow.js](https://www.tensorflow.org/js)

## Estrutura do projeto

```
.
├── index.html
├── style.css
└── script.js
```

## Como executar

Clone o repositório e abra o arquivo `index.html` diretamente no navegador. Uma conexão com a internet é necessária para carregar o TensorFlow.js e as fontes via CDN.

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

Não há dependências para instalar nem build necessário.

## Como funciona

O modelo é uma regressão linear implementada com uma rede neural de camada única (1 neurônio). A cada execução, ele passa pelo seguinte fluxo:

1. Recebe a quantidade digitada pelo usuário
2. Cria e compila o modelo com `meanSquaredError` como função de perda e `sgd` como otimizador
3. Treina por 200 épocas com os dados de exemplo
4. Realiza a previsão e exibe o preço estimado

Os dados de treino seguem uma relação linear direta (R$ 15 por item), o que permite ao modelo convergir com alta precisão em poucas épocas.

## Deploy

O projeto está hospedado via GitHub Pages e pode ser acessado em:

```
https://kunglaogaucho.github.io/ML_umc/
```
