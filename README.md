# 👨‍💻 Sistema de Reconhecimento Facial com OpenCV

Este é um projeto prático de **Visão Computacional** desenvolvido em Python. O sistema utiliza a biblioteca OpenCV para realizar a detecção e o reconhecimento facial em tempo real através da webcam, construindo o próprio dataset e treinando o modelo de Machine Learning dinamicamente.

## 🚀 Funcionalidades

- **Detecção em Tempo Real:** Identifica rostos no feed da webcam usando o classificador Haar Cascade.
- **Criação de Dataset Dinâmico:** Permite cadastrar novos usuários capturando fotos (recortes do rosto em tons de cinza) diretamente pela câmera.
- **Treinamento Automático:** Utiliza o algoritmo LBPH (Local Binary Patterns Histograms) para treinar a Inteligência Artificial com as fotos coletadas.
- **Múltiplos Usuários:** Capacidade de cadastrar e reconhecer diferentes pessoas, salvando o estado do aprendizado em um arquivo `.yml`.
- **Interface Interativa:** Menu no terminal para alternar entre o modo de *Cadastro de Rostos* e o modo de *Reconhecimento*.

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **OpenCV** (`opencv-python` e `opencv-contrib-python`) - Para processamento de imagens e algoritmos de visão computacional.
- **NumPy** - Para manipulação de arrays e dados matemáticos necessários no treinamento do modelo.

## ⚙️ Como executar o projeto

### Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina. Em seguida, instale as dependências executando o comando abaixo no terminal:

```bash
pip install opencv-contrib-python numpy

```

### Rodando a aplicação

1. Faça o download ou clone este repositório.
2. Abra o terminal na pasta do projeto e execute o script:

```bash
python seu_arquivo_de_codigo.py

```

*(Substitua `seu_arquivo_de_codigo.py` pelo nome exato do seu arquivo)*

3. **Modo de Cadastro (Opção 2):** Ao rodar pela primeira vez, escolha a opção 2 para registrar o seu rosto. Pressione a tecla `c` 10 vezes com a janela do vídeo selecionada para capturar as amostras.
4. **Modo de Reconhecimento (Opção 1):** Após o cadastro, o modelo será treinado automaticamente. Escolha a opção 1 para iniciar a câmera e testar o reconhecimento!

## 📂 Estrutura de Arquivos

* `codigo.py`: Script principal contendo toda a lógica de detecção, cadastro e reconhecimento.
* `dataset_rostos/`: Pasta gerada automaticamente pelo código para armazenar as fotos capturadas. *(Nota: Não inclusa no repositório por questões de privacidade).*
* `modelo_lbph.yml`: Arquivo gerado após o treinamento, que armazena a "memória" da rede neural. *(Nota: Não incluso no repositório por questões de privacidade).*

## 💡 Próximos Passos (Melhorias Futuras)

* [ ] Exibir a porcentagem de confiança/certeza da IA na tela do vídeo.
* [ ] Criar um arquivo de log (TXT) registrando os horários de acesso de usuários conhecidos.
* [ ] Implementar detecção de "Invasores" com alertas.

---

Desenvolvido por **[Charles Moraes Rodrigues]**
Conecte-se comigo no [LinkedIn] www.linkedin.com/in/charles-moraes-rodrigues-06250967




```

