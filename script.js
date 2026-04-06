// Dados de treino: quantidade de itens x preço total em reais
// O modelo vai aprender que cada item custa R$ 15
const DADOS_QUANTIDADE = [1, 2, 3, 4, 5, 6];
const DADOS_PRECO      = [15, 30, 45, 60, 75, 90];
const EPOCHS           = 200;

// Referências aos elementos da interface
const ui = {
  quantidade:   document.getElementById("quantidade"),
  status:       document.getElementById("status"),
  dot:          document.getElementById("dot"),
  resultado:    document.getElementById("resultado"),
  resultadoBox: document.getElementById("resultadoBox"),
};

function atualizarStatus(mensagem, estado) {
  ui.status.textContent = mensagem;
  ui.dot.className = `dot ${estado}`; // estados: '', 'ativo', 'pronto'
}

function criarModelo() {
  // Rede neural simples com 1 neurônio: 1 entrada (quantidade) → 1 saída (preço)
  const modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // meanSquaredError mede o erro; sgd ajusta os pesos a cada epoch para reduzi-lo
  modelo.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  return modelo;
}

async function treinar(modelo) {
  const xs = tf.tensor2d(DADOS_QUANTIDADE, [DADOS_QUANTIDADE.length, 1]);
  const ys = tf.tensor2d(DADOS_PRECO, [DADOS_PRECO.length, 1]);

  await modelo.fit(xs, ys, { epochs: EPOCHS });

  // Libera os tensores de treino da memória após o uso
  xs.dispose();
  ys.dispose();
}

function prever(modelo, quantidade) {
  const input    = tf.tensor2d([quantidade], [1, 1]);
  const output   = modelo.predict(input);
  const resultado = output.dataSync()[0];

  // Libera os tensores de previsão da memória após o uso
  input.dispose();
  output.dispose();

  return resultado;
}

function exibirResultado(preco) {
  ui.resultado.textContent = `R$ ${preco.toFixed(2)}`;
  ui.resultadoBox.classList.add("visivel");
}

window.treinarEPrever = async function () {
  const quantidade = Number(ui.quantidade.value);

  if (!quantidade || quantidade <= 0) {
    atualizarStatus("Digite uma quantidade válida.", "");
    return;
  }

  ui.resultadoBox.classList.remove("visivel");
  atualizarStatus("Treinando o modelo...", "ativo");

  const modelo = criarModelo();
  await treinar(modelo);

  atualizarStatus("Modelo treinado! Calculando previsão...", "ativo");

  const precoPrevisto = prever(modelo, quantidade);

  // Libera o modelo da memória após a previsão
  modelo.dispose();

  atualizarStatus("Previsão concluída!", "pronto");
  exibirResultado(precoPrevisto);
};