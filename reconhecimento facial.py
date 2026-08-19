
import cv2
import numpy as np
import os

# 1. Configurações iniciais
pasta_base = os.path.dirname(os.path.abspath(__file__))
pasta_fotos = os.path.join(pasta_base, "dataset_rostos")
arquivo_modelo = os.path.join(pasta_base, "modelo_lbph.yml")
os.makedirs(pasta_fotos, exist_ok=True)

caminho_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
classificador_rosto = cv2.CascadeClassifier(caminho_cascade)
reconhecedor = cv2.face.LBPHFaceRecognizer_create()

# 2. LISTA DE NOMES (O índice da lista corresponde ao ID)
# ID 0 = Desconhecido | ID 1 = Você | ID 2 = Amigo | ID 3 = Familiar
nomes = ["Desconhecido", "Seu nome", "Nome do Amigo", "Nome do Familiar"] 

# 3. MENU INTERATIVO NO TERMINAL
print("--- SISTEMA MULTI-FACES ---")
print("1 - Iniciar Reconhecimento (Câmera)")
print("2 - Cadastrar novo rosto")
try:
    opcao = input("Escolha uma opção (Digite 1 ou 2 e aperte Enter): ").strip()
except EOFError:
    print("Nenhuma opção foi informada. Execute o arquivo em um terminal e digite 1 ou 2.")
    opcao = ""

if opcao == '2':
    # --- MODO DE CADASTRO ---
    print(f"\nNomes disponíveis:")
    for i in range(1, len(nomes)):
        print(f"ID {i}: {nomes[i]}")
        
    id_atual = int(input("\nDigite o ID numérico da pessoa que vai para a frente da câmera: "))
    
    fotos_coletadas = 0
    max_fotos = 10
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("Erro: não foi possível abrir a câmera. Feche outros programas que possam estar usando-a.")
        exit()
    print(f"\nPressione 'c' para capturar as fotos (0/{max_fotos}).")
    
    while True:
        sucesso, frame = camera.read()
        if not sucesso: break
        
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = classificador_rosto.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in rostos:
            cv2.putText(frame, f"Aperte 'c': {fotos_coletadas}/{max_fotos}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        cv2.imshow("Modo Cadastro", frame)
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord('q'): break
        elif tecla == ord('c'):
            if len(rostos) == 1:
                (x, y, w, h) = rostos[0]
                rosto_recortado = frame_cinza[y:y+h, x:x+w]
                rosto_padrao = cv2.resize(rosto_recortado, (200, 200))
                
                fotos_coletadas += 1
                # Salva com o padrão: usuario.ID.FOTO.jpg
                nome_arquivo = f"{pasta_fotos}/usuario.{id_atual}.{fotos_coletadas}.jpg"
                cv2.imwrite(nome_arquivo, rosto_padrao)
                print(f"Foto salva: {nome_arquivo}")
                
                if fotos_coletadas == max_fotos:
                    break
            else:
                print("Aviso: Mantenha exatamente 1 rosto na tela.")
                
    camera.release()
    cv2.destroyAllWindows()
    
    # 4. ROTINA DE TREINAMENTO COM TODAS AS FOTOS DA PASTA
    if fotos_coletadas == max_fotos:
        print("\nTreinando a IA com TODAS as fotos salvas na pasta...")
        # Pega o caminho de todas as imagens .jpg na pasta
        caminhos = [os.path.join(pasta_fotos, f) for f in os.listdir(pasta_fotos) if f.endswith('.jpg')]
        rostos_treinamento = []
        ids_treinamento = []
        
        for caminho in caminhos:
            # Carrega a imagem
            imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            # Extrai o ID do nome do arquivo (ex: em usuario.2.1.jpg, ele pega o '2')
            id_img = int(os.path.split(caminho)[-1].split('.')[1])
            
            rostos_treinamento.append(imagem)
            ids_treinamento.append(id_img)
            
        # Treina e salva o modelo
        reconhecedor.train(rostos_treinamento, np.array(ids_treinamento, dtype=np.int32))
        reconhecedor.write(arquivo_modelo)
        print("Treinamento concluído com sucesso! Rode o script novamente e escolha a Opção 1.")

elif opcao == '1':
    # --- MODO DE RECONHECIMENTO ---
    if not os.path.exists(arquivo_modelo):
        print("\nErro: Modelo não encontrado. Rode o script e escolha a Opção 2 para cadastrar alguém primeiro.")
        exit()
        
    reconhecedor.read(arquivo_modelo)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("Erro: não foi possível abrir a câmera. Feche outros programas que possam estar usando-a.")
        exit()
    print("\nIniciando reconhecimento... Pressione 'q' na janela do vídeo para sair.")
    
    while True:
        sucesso, frame = camera.read()
        if not sucesso: break
        
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = classificador_rosto.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in rostos:
            rosto_recortado = frame_cinza[y:y+h, x:x+w]
            rosto_padrao = cv2.resize(rosto_recortado, (200, 200))
            
            # Previsão da IA
            id_previsto, confianca = reconhecedor.predict(rosto_padrao)
            
            # Verifica se a confiança é boa e se o ID existe na nossa lista de nomes
            if confianca < 70 and id_previsto < len(nomes):
                nome = nomes[id_previsto]
                cor = (0, 255, 0) # Verde
            else:
                nome = nomes[0] # Desconhecido
                cor = (0, 0, 255) # Vermelho
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(frame, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            
        cv2.imshow("Reconhecimento Multi-Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    camera.release()
    cv2.destroyAllWindows()

else:
    print("Opção inválida. Rode o script novamente.")
    