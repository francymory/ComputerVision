# ComputerVision

To start the Unimore GPU with a Jupyter notebook:
1. execute ./start_jupyter.sh and copy the link and change the permissions
2. on the notebook select an existing jupyter notebook, past the link and choose python
3. now you can execute the notebook cells

Add a pytorch-CycleGAN-and-pix2pix directory with the models from github
1. in your project directory: git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
2. cd pytorch-CycleGAN-and-pix2pix
3. pip install -r requirements.txt

Final preprocessed paired dataset:
/work/cvcs2024/fiorottandi/workspace/data/dinov2_dataset/combinedAB   -> obtained with dinov2 as retrieval model
/work/cvcs2024/fiorottandi/workspace/data/dataset/combinedAB -> obtained with resnet18 (less precise)

To train a pretrained pix2pix model (day2night), dentro la directory pytorch-CycleGAN-and-pix2pix a linea di comando:
1. bash ./scripts/download_pix2pix_model.sh day2night  (Questo comando scaricherà il modello preaddestrato nella cartella ./checkpoints/day2night_pretrained/ con all'interno il checkpoint del generatore)
2. Nel file base_model.py, nel metodo load_networks, aggiungi un controllo per verificare se il file del discriminatore esiste. Se non esiste, salta il caricamento di latest_net_D.pth (che nel modello pretrainato non esiste e darebbe errore). il discriminatore verrà inizializzato così con pesi casuali.

#code:
def load_networks(self, epoch):
    """Load all the networks from the disk.
    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    for name in self.model_names:
        if isinstance(name, str):
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            
            # Aggiunta del controllo per l'esistenza del file -----> QUESTO
            if not os.path.exists(load_path):
                if name == 'D':  # Se è il discriminatore, mostra un messaggio e continua
                    print(f"File {load_path} does not exist. Initializing the discriminator from scratch.")
                    continue
                else:
                    raise FileNotFoundError(f"File {load_path} does not exist and is required.")
            
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)

3. srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --account=cvcs2024 --time 120:00 --pty python train.py --dataroot /work/cvcs2024/fiorottandi/workspace/data/dinov2_dataset/combinedAB --name day2night_pretrained --model pix2pix --direction AtoB --epoch_count 1 --n_epochs 25 --n_epochs_decay 25
4. Per continuare ad allenare da un checkpoint:
   1. Commenta il codice aggiunto prima per ignorare il discriminatore (ora c'è e lo vuoi considerare)
   2. crea una nuova directory  (es: ./checkpoints/day2night_pretrained_2)  e copiaci dentro solo latest_net_G.pth e latest_net_D.pth della directory precedente (se il     
    training si è interrotto bruscamente prendi l'ultimo checkpoint valido (es 40_net_G.pth e 40_net_D.pth), copia nella nuova directory e rinomina latest_net_G.pth e 
    latest_net_D.pth
   3. allena come prima tenendo epoch_count 1 e cambiando solo --name directory: srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --account=cvcs2024 --time 120:00 --pty python train.py --dataroot /work/cvcs2024/fiorottandi/workspace/data/dinov2_dataset/combinedAB --name day2night_pretrained_2 --model pix2pix --direction AtoB --epoch_count 1 --n_epochs 25 --n_epochs_decay 25


