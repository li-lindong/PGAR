from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from infer_module.dynamic_infer_module import Dynamic_Person_Inference, Hierarchical_Dynamic_Inference, Multi_Dynamic_Inference
from infer_module.pctdm_infer_module import PCTDM
from infer_module.higcin_infer_module import CrossInferBlock
from infer_module.AT_infer_module import Actor_Transformer, Embfeature_PositionEmbedding
from infer_module.ARG_infer_module import GCN_Module
from infer_module.SACRF_BiUTE_infer_module import SACRF, BiUTE
from infer_module.TCE_STBiP_module import MultiHeadLayerEmbfeatureContextEncoding
from infer_module.positional_encoding import Context_PositionEmbeddingSine
import collections
from torch_geometric.nn import GatedGraphConv, GCNConv, GATConv


class Dynamic_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        
        
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape = (10, 12),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                num_DIM = cfg.num_DIM,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)


        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features, inplace = True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace = True)
        else:
            None

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()


        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace = True)
            boxes_states = self.dropout_global(boxes_states)


        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states,dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B,-1)

        return {'activities':activities_scores} # actions_scores, activities_scores


class Dynamic_TCE_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_TCE_volleyball, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_features_context = 128

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        # TCE Module Loading
        self.multilayer_head_embfeature_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context, 1,
                num_features_context, NFB, K, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16, 512 / 2)

        # DIN
        context_dim = in_dim + num_heads_context * num_features_context
        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim=context_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                num_DIM=cfg.num_DIM,
                cfg=cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim=context_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg=cfg, )
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, context_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(context_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace=True)
        else:
            None

        # Context Positional Encoding
        context = outputs[-1]
        context = self.context_positionembedding1(context)
        # Embedded Feature Context Encoding
        context_states = self.multilayer_head_embfeature_context_encoding(boxes_features, context)
        context_states = context_states.reshape(B, T, N, -1)
        boxes_features = torch.cat((boxes_features, context_states), dim=3)

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()

        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)

        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num


        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities': activities_scores}  # actions_scores, activities_scores



class PCTDM_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(PCTDM_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.pctdm = PCTDM(cfg)
        self.pctdm_nl = nn.LayerNorm([T, 2000])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        self.fc_activities = nn.Linear(2000, self.cfg.num_activities)
        self.fc_actions = nn.Linear(2000, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes


        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        # PCTDM inference
        graph_boxes_features = self.pctdm(boxes_features)
        torch.cuda.empty_cache()

        boxes_states = graph_boxes_features.view((B, T, -1))
        boxes_states = self.pctdm_nl(boxes_states)
        boxes_states = F.relu(boxes_states, inplace=True)
        boxes_states = self.dropout_global(boxes_states)

        # Predict actions
        # actions_scores = self.fc_actions(boxes_states)
        # actions_scores = actions_scores.view((B, T, N, -1))
        # actions_scores = torch.mean(actions_scores, dim = 1)
        # actions_scores = actions_scores.view(B*N, -1)

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
        boxes_states_pooled_flat = boxes_states.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)


        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)

        # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        # boxes_states_pooled_flat = boxes_states
        # activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        # activities_scores = activities_scores.reshape(B, T, -1)
        # activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities':activities_scores} # [actions_scores, activities_scores] # activities_scores


class HiGCIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(HiGCIN_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.person_avg_pool = nn.AvgPool2d((K**2, 1), stride = 1)
        self.BIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = K**2)
        self.PIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = N)
        self.dropout = nn.Dropout()
        self.fc_activities = nn.Linear(D, cfg.num_activities, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]


        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.view(B, T, N, D, K*K)
        boxes_features = boxes_features.permute(0, 2, 1, 4, 3).contiguous()
        boxes_features = boxes_features.view(B*N, T, K*K, D) # B*N, T, K*K, D

        # HiGCIN Inference
        boxes_features = self.BIM(boxes_features) # B*N, T, K*K, D
        boxes_features = self.person_avg_pool(boxes_features) # B*N, T, D
        boxes_features = boxes_features.view(B, N, T, D).contiguous().permute(0, 2, 1, 3) # B, T, N, D
        boxes_states = self.PIM(boxes_features) # B, T, N, D
        boxes_states = self.dropout(boxes_states)
        torch.cuda.empty_cache()

        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities':activities_scores}


class AT_volleyball(nn.Module):
    def __init__(self, cfg):
        super(AT_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # AT inference
        self.PE = Embfeature_PositionEmbedding(cfg = cfg, num_pos_feats = NFB//2)
        self.AT = Actor_Transformer(in_dim = NFB, temporal_pooled_first = cfg.temporal_pooled_first)
        self.fc_activities = nn.Linear(NFB, cfg.num_activities)
        self.fc_actions = nn.Linear(NFB, cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        # AT inference
        boxes_features = self.PE(boxes_features, boxes_in_flat)
        boxes_states = self.AT(boxes_features)
        torch.cuda.empty_cache()

        if self.cfg.temporal_pooled_first:
            # Predict actions
            actions_scores = self.fc_actions(boxes_states)
            actions_scores = actions_scores.view(B * N, -1)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
            boxes_states_pooled_flat = boxes_states_pooled.reshape(B, -1)
            activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
        else:
            # Predict actions
            actions_scores = self.fc_actions(boxes_states)
            actions_scores = actions_scores.view((B, T, N, -1))
            actions_scores = torch.mean(actions_scores, dim = 1)
            actions_scores = actions_scores.view(B*N, -1)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=1)
            boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
            activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num
            activities_scores = activities_scores.reshape(B, T, -1)
            activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        # # Predict actions
        # boxes_states_flat = boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores = self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        # # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)

        # return {'activities':activities_scores, 'actions':actions_scores, 'boxes_states':boxes_states.view((B, T, N, -1))} # [actions_scores, activities_scores] # activities_scores
        return {'activities':activities_scores}


class ARG_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(ARG_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph


        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if not self.training:
            B = B * 3
            T = T // 3
            images_in.reshape((B, T) + images_in.shape[2:])
            boxes_in.reshape((B, T) + boxes_in.shape[2:])

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features.reshape(B, T * N, NFG)

        #         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)
        #             visual_info.append(relation_graph.reshape(B,T,N,N))

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = boxes_features.reshape(B, T, N, NFB)

        #         boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features

        boxes_states = self.dropout_global(boxes_states)

        NFS = NFG

        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        # actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B, T, N, -1)
        # actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        if not self.training:
            B = B // 3
            # actions_scores = torch.mean(actions_scores.reshape(B, 3, N, -1), dim=1).reshape(B * N, -1)
            activities_scores = torch.mean(activities_scores.reshape(B, 3, -1), dim=1).reshape(B, -1)

        # return [activities_scores] # actions_scores, #'boxes_states':boxes_states
        # return {'activities':activities_scores, 'actions_scores':actions_scores}
        return {'activities':activities_scores}


class SACRF_BiUTE_volleyball(nn.Module):
    def __init__(self, cfg):
        super(SACRF_BiUTE_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # SACRF+BiUTE inference
        self.SACRF = SACRF(cfg, NFB, num_actions = cfg.num_actions)
        self.BiUTE = BiUTE(NFB, cfg.num_boxes)
        self.fc_activities = nn.Linear(NFB*2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)


        # AT inference
        action_scores, C_v, halt_loss = self.SACRF(boxes_features, boxes_in_flat)
        activities_feature = self.BiUTE(C_v)
        activities_scores = self.fc_activities(activities_feature)
        torch.cuda.empty_cache()

        action_scores = torch.mean(action_scores, dim = 1).view(B*N, -1)
        activities_scores = torch.mean(activities_scores, dim = 1).view(B, -1)

        return {'actions': action_scores, 'activities':activities_scores, 'halting':halt_loss, 'boxes_states':C_v}
        # return [actions_scores, activities_scores]  # activities_scores



class Dynamic_collective(nn.Module):
    def __init__(self, cfg):
        super(Dynamic_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        #self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            self.DPI = Dynamic_Person_Inference(
                in_dim = in_dim,
                person_mat_shape = (T, N),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(T, N),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        if self.cfg.lite_dim:
            boxes_features_all = boxes_features_all.permute(0, 3, 1, 2)
            boxes_features_all = self.point_conv(boxes_features_all)
            boxes_features_all = boxes_features_all.permute(0, 2, 3, 1)
            boxes_features_all = self.point_ln(boxes_features_all)
            boxes_features_all = F.relu(boxes_features_all, inplace = True)
        else:
            None

        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 4)

        #actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4

            # Dynamic graph inference
            graph_boxes_features = self.DPI(boxes_features)
            torch.cuda.empty_cache()

            # cat graph_boxes_features with boxes_features
            boxes_states = graph_boxes_features + boxes_features  # 1, T, N, NFG
            boxes_states = boxes_states.permute(0, 2, 1, 3).view(N, T, -1)
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            # boxes_states = boxes_states.view(T, N, -1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim = 0)  # T, NFS
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return {'activities':activities_scores}# activities_scores # actions_scores,

class bb_ggnn_edge_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_ggnn_edge_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)

        self.fc_activities = nn.Linear(NFB * 2, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4

            # 产生全连接图，并做图推理
            edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                edge_weight = edge_weight.cuda()
                edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index, edge_weight)))

            edge_weight_T = torch.stack(edge_weight_T, dim=0)
            edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            group_features = group_features_rgb

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            activities_features.append(group_features)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        activities_features = torch.cat(activities_features, dim=0)

        return {'activities': activities_scores, 'edge_weight': edge_weight_all, 'activities_features': activities_features}# activities_scores # actions_scores,

class bb_ff_skeleton_trans_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_ff_skeleton_trans_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        print("冻结ff_skeleton_trans模型参数！！")
        for m in zip(self.pose_embedding.parameters(), self.ori_embedding.parameters(), self.pose_embedding.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.gnn_pos.parameters(), self.gnn_ori.parameters(), self.gnn_pose.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.ind_transformer.parameters(), self.fc_fformation.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)
        state['fc_fformation.weight'] = state['fc_activity.weight']
        state['fc_fformation.bias'] = state['fc_activity.bias']
        del state['fc_activity.weight']
        del state['fc_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, MAX_N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                # edge_index, _ = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                # edge_index = edge_index.cuda()
                # boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            # boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            group_features_ff = torch.mean(group_features_ff, dim=0).reshape(1, -1)
            activities_features.append(group_features)
            activities_features_ff.append(group_features_ff)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        activities_features = torch.cat(activities_features, dim=0)
        activities_features_ff = torch.cat(activities_features_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff,
                'activities_features': activities_features, 'activities_features_ff': activities_features_ff}# activities_scores # actions_scores,

class bb_ggnn_edge_ff_skeleton_trans_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_ggnn_edge_ff_skeleton_trans_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        print("冻结ff_skeleton_trans模型参数！！")
        for m in zip(self.pose_embedding.parameters(), self.ori_embedding.parameters(), self.pose_embedding.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.gnn_pos.parameters(), self.gnn_ori.parameters(), self.gnn_pose.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.ind_transformer.parameters(), self.fc_fformation.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)
        state['fc_fformation.weight'] = state['fc_activity.weight']
        state['fc_fformation.bias'] = state['fc_activity.bias']
        del state['fc_activity.weight']
        del state['fc_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, MAX_N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                edge_weight = edge_weight.cuda()
                edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index, edge_weight)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            edge_weight_T = torch.stack(edge_weight_T, dim=0)
            edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            group_features_ff = torch.mean(group_features_ff, dim=0).reshape(1, -1)
            activities_features.append(group_features)
            activities_features_ff.append(group_features_ff)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        activities_features = torch.cat(activities_features, dim=0)
        activities_features_ff = torch.cat(activities_features_ff, dim=0)

        return {'activities': activities_scores, 'edge_weight': edge_weight_all, 'activities_ff': activities_scores_ff,
                'activities_features': activities_features, 'activities_features_ff': activities_features_ff}# activities_scores # actions_scores,

class bb_ggnn_ff_skeleton_trans_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_ggnn_ff_skeleton_trans_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        print("冻结ff_skeleton_trans模型参数！！")
        for m in zip(self.pose_embedding.parameters(), self.ori_embedding.parameters(), self.pose_embedding.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.gnn_pos.parameters(), self.gnn_ori.parameters(), self.gnn_pose.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.ind_transformer.parameters(), self.fc_fformation.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)
        state['fc_fformation.weight'] = state['fc_activity.weight']
        state['fc_fformation.bias'] = state['fc_activity.bias']
        del state['fc_activity.weight']
        del state['fc_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, MAX_N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, _ = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            group_features_ff = torch.mean(group_features_ff, dim=0).reshape(1, -1)
            activities_features.append(group_features)
            activities_features_ff.append(group_features_ff)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        activities_features = torch.cat(activities_features, dim=0)
        activities_features_ff = torch.cat(activities_features_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff,
                'activities_features': activities_features, 'activities_features_ff': activities_features_ff}# activities_scores # actions_scores,

class bb_gat_ff_skeleton_trans_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_gat_ff_skeleton_trans_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.ggnn = GatedGraphConv(NFB, num_layers=3)
        self.gat = GATConv(in_channels=NFB, out_channels=NFB)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        print("冻结ff_skeleton_trans模型参数！！")
        for m in zip(self.pose_embedding.parameters(), self.ori_embedding.parameters(), self.pose_embedding.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.gnn_pos.parameters(), self.gnn_ori.parameters(), self.gnn_pose.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.ind_transformer.parameters(), self.fc_fformation.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)
        state['fc_fformation.weight'] = state['fc_activity.weight']
        state['fc_fformation.bias'] = state['fc_activity.bias']
        del state['fc_activity.weight']
        del state['fc_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, MAX_N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                # edge_weight = edge_weight.cuda()
                # edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.gat(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # edge_weight_T = torch.stack(edge_weight_T, dim=0)
            # edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            group_features_ff = torch.mean(group_features_ff, dim=0).reshape(1, -1)
            activities_features.append(group_features)
            activities_features_ff.append(group_features_ff)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        activities_features = torch.cat(activities_features, dim=0)
        activities_features_ff = torch.cat(activities_features_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff,
                'activities_features': activities_features, 'activities_features_ff': activities_features_ff}# activities_scores # actions_scores,

class bb_gcn_ff_skeleton_trans_collective(nn.Module):
    def __init__(self, cfg):
        super(bb_gcn_ff_skeleton_trans_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
            print("初始化backbone为inceptionv3！")
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
            print("初始化backbone为vgg16！")
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
            print("初始化backbone为vgg19！")
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
            print("初始化backbone为Res18！")
        else:
            assert False

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.ggnn = GatedGraphConv(NFB, num_layers=3)
        # self.gat = GATConv(in_channels=NFB, out_channels=NFB)
        self.gcn = GCNConv(in_channels=NFB, out_channels=NFB)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        print("冻结ff_skeleton_trans模型参数！！")
        for m in zip(self.pose_embedding.parameters(), self.ori_embedding.parameters(), self.pose_embedding.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.gnn_pos.parameters(), self.gnn_ori.parameters(), self.gnn_pose.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False
            m[2].requires_grad = False
        for m in zip(self.ind_transformer.parameters(), self.fc_fformation.parameters()):
            m[0].requires_grad = False
            m[1].requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)
        state['fc_fformation.weight'] = state['fc_activity.weight']
        state['fc_fformation.bias'] = state['fc_activity.bias']
        del state['fc_activity.weight']
        del state['fc_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, MAX_N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        #actions_scores = []
        activities_scores = []
        activities_features = []
        activities_scores_ff = []
        activities_features_ff = []
        edge_weight_all = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # 1,T,N,NFB
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                # edge_weight = edge_weight.cuda()
                # edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.gcn(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # edge_weight_T = torch.stack(edge_weight_T, dim=0)
            # edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

            group_features = torch.mean(group_features, dim=0).reshape(1, -1)
            group_features_ff = torch.mean(group_features_ff, dim=0).reshape(1, -1)
            activities_features.append(group_features)
            activities_features_ff.append(group_features_ff)


        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        activities_features = torch.cat(activities_features, dim=0)
        activities_features_ff = torch.cat(activities_features_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff,
                'activities_features': activities_features, 'activities_features_ff': activities_features_ff}# activities_scores # actions_scores,

class bb_ggnn_edge_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(bb_ggnn_edge_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)

        self.fc_activities = nn.Linear(NFB * 2, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)['state_dict']
        state['fc_fformation.weight'] = state['classifier_activity.weight']
        state['fc_fformation.bias'] = state['classifier_activity.bias']
        del state['classifier_activity.weight']
        del state['classifier_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        activities_scores = []
        edge_weight_all = []
        for b in range(B):
            boxes_features = boxes_features_all[b, :, :N, :]

            # 产生全连接图，并做图推理
            edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                edge_weight = edge_weight.cuda()
                edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index, edge_weight)))

            edge_weight_T = torch.stack(edge_weight_T, dim=0)
            edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            group_features = group_features_rgb

            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return {'activities': activities_scores, 'edge_weight': edge_weight_all}  # actions_scores, activities_scores

class bb_ggnn_edge_ff_skeleton_trans_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(bb_ggnn_edge_ff_skeleton_trans_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)['state_dict']
        state['fc_fformation.weight'] = state['classifier_activity.weight']
        state['fc_fformation.bias'] = state['classifier_activity.bias']
        del state['classifier_activity.weight']
        del state['classifier_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        activities_scores = []
        activities_scores_ff = []
        edge_weight_all = []
        for b in range(B):
            boxes_features = boxes_features_all[b, :, :N, :]
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                edge_weight = edge_weight.cuda()
                edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index, edge_weight)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            edge_weight_T = torch.stack(edge_weight_T, dim=0)
            edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        return {'activities': activities_scores, 'edge_weight': edge_weight_all, 'activities_ff': activities_scores_ff}  # actions_scores, activities_scores

class bb_gcn_ff_skeleton_trans_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(bb_gcn_ff_skeleton_trans_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.ggnn = GatedGraphConv(NFB, num_layers=3)
        # self.gat = GATConv(in_channels=NFB, out_channels=NFB)
        self.gcn = GCNConv(in_channels=NFB, out_channels=NFB)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)['state_dict']
        state['fc_fformation.weight'] = state['classifier_activity.weight']
        state['fc_fformation.bias'] = state['classifier_activity.bias']
        del state['classifier_activity.weight']
        del state['classifier_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        activities_scores = []
        activities_scores_ff = []
        edge_weight_all = []
        for b in range(B):
            boxes_features = boxes_features_all[b, :, :N, :]
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                # edge_weight = edge_weight.cuda()
                # edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.gcn(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # edge_weight_T = torch.stack(edge_weight_T, dim=0)
            # edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff}  # actions_scores, activities_scores

class bb_gat_ff_skeleton_trans_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(bb_gat_ff_skeleton_trans_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.ggnn = GatedGraphConv(NFB, num_layers=3)
        self.gat = GATConv(in_channels=NFB, out_channels=NFB)
        # self.gcn = GCNConv(in_channels=NFB, out_channels=NFB)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)['state_dict']
        state['fc_fformation.weight'] = state['classifier_activity.weight']
        state['fc_fformation.bias'] = state['classifier_activity.bias']
        del state['classifier_activity.weight']
        del state['classifier_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        activities_scores = []
        activities_scores_ff = []
        edge_weight_all = []
        for b in range(B):
            boxes_features = boxes_features_all[b, :, :N, :]
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                # edge_weight = edge_weight.cuda()
                # edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.gat(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # edge_weight_T = torch.stack(edge_weight_T, dim=0)
            # edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff}  # actions_scores, activities_scores

class bb_ggnn_ff_skeleton_trans_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(bb_ggnn_ff_skeleton_trans_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.ggnn = GatedGraphConv(NFB, num_layers=3)
        # self.gat = GATConv(in_channels=NFB, out_channels=NFB)
        # self.gcn = GCNConv(in_channels=NFB, out_channels=NFB)

        self.pos_embedding = nn.Sequential(nn.Linear(2, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.ori_embedding = nn.Sequential(nn.Linear(8, 256),
                                           nn.ReLU(),
                                           nn.Dropout())
        self.pose_embedding = nn.Sequential(nn.Linear(34, 256),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.gnn_pos = GCNConv(in_channels=256, out_channels=256)
        self.gnn_ori = GCNConv(in_channels=256, out_channels=256)
        self.gnn_pose = GCNConv(in_channels=256, out_channels=256)

        self.ind_transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=3)

        self.fc_activities = nn.Linear(NFB * 2 + 256 * 3 * 2, self.cfg.num_activities)
        self.fc_fformation = nn.Linear(256 * 3 * 2, cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def load_ff_skeleton_trans_model(self, filepath):
        state = torch.load(filepath)['state_dict']
        state['fc_fformation.weight'] = state['classifier_activity.weight']
        state['fc_fformation.bias'] = state['classifier_activity.bias']
        del state['classifier_activity.weight']
        del state['classifier_activity.bias']
        model_dict = self.state_dict()
        model_dict.update(state)
        self.load_state_dict(model_dict)
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, orientations_in, positions_in, skeletons_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        # assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        # Embedding F-Formation
        skeletons_in = torch.reshape(skeletons_in, [B, T, N, -1])
        orientations_embedding = self.ori_embedding(orientations_in)
        positions_embedding = self.pos_embedding(positions_in)
        skeletons_embedding = self.pose_embedding(skeletons_in)

        activities_scores = []
        activities_scores_ff = []
        edge_weight_all = []
        for b in range(B):
            boxes_features = boxes_features_all[b, :, :N, :]
            orientations_embedding_b = orientations_embedding[b, :, :N, :]
            positions_embedding_b = positions_embedding[b, :, :N, :]
            skeletons_embedding_b = skeletons_embedding[b, :, :N, :]

            # 产生全连接图，并做图推理
            # edge_weight_T = []  # 存储一个样本每一帧中的edge_weight
            boxes_features_graph = []
            ff_features_graph_trans = []
            boxes_features = torch.squeeze(boxes_features, dim=0)
            for t in range(T):

                # edge_index, edge_weight = generate_graph(N, boxes_features[t], self_loop=True)  # 此处计算比较耗时，有空可优化
                edge_index, edge_weight = create_edge_index_and_weight(boxes_features[t], self_loop=True)  # 优化后的代码
                edge_index = edge_index.cuda()
                # edge_weight = edge_weight.cuda()
                # edge_weight_T.append(edge_weight)
                boxes_features_graph.append(F.relu(self.ggnn(boxes_features[t], edge_index)))

                edge_index_ori, edge_index_pos, edge_index_ske = create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False), create_edge_index(N, self_loop=False)
                edge_index_ori = edge_index_ori.cuda()
                edge_index_pos = edge_index_pos.cuda()
                edge_index_ske = edge_index_ske.cuda()
                ori_feature = F.relu(self.gnn_ori(orientations_embedding_b[t], edge_index_ori))
                pos_feature = F.relu(self.gnn_pos(positions_embedding_b[t], edge_index_pos))
                ske_feature = F.relu(self.gnn_pose(skeletons_embedding_b[t], edge_index_ske))
                ff_feature = torch.stack([ori_feature, pos_feature, ske_feature], dim=-2).reshape(N, 3, -1)
                ff_feature_trans = self.ind_transformer.encoder(ff_feature)
                ff_feature_trans_residual = torch.cat([ff_feature, ff_feature_trans], dim=-1).reshape(N, -1)
                ff_features_graph_trans.append(ff_feature_trans_residual)

            # edge_weight_T = torch.stack(edge_weight_T, dim=0)
            # edge_weight_all.append(edge_weight_T)
            boxes_features_graph = torch.stack(boxes_features_graph, dim=0)
            boxes_features_graph_residual = torch.cat([boxes_features, boxes_features_graph], dim=-1)
            group_features_rgb = torch.mean(boxes_features_graph_residual, dim=1)

            ff_features_graph_trans = torch.stack(ff_features_graph_trans, dim=0)
            group_features_ff = torch.mean(ff_features_graph_trans, dim=1)

            group_features = torch.cat([group_features_rgb, group_features_ff], dim=1)

            acty_score = self.fc_activities(group_features)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)
            acty_score_ff = self.fc_fformation(group_features_ff)
            acty_score_ff = torch.mean(acty_score_ff, dim=0).reshape(1, -1)
            activities_scores_ff.append(acty_score_ff)

        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        activities_scores_ff = torch.cat(activities_scores_ff, dim=0)

        return {'activities': activities_scores, 'activities_ff': activities_scores_ff}  # actions_scores, activities_scores

def generate_graph(rel_num, vectors, self_loop):
    numNode = rel_num
    edge_index, edge_weight = [], []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if self_loop:
                    edge_index.append([i, j])
                    edge_weight.append((cos_sim(vectors[i], vectors[j]) + 1) / 2.0)
                else:
                    if i != j:
                        edge_index.append([i, j])
                        edge_weight.append((cos_sim(vectors[i], vectors[j]) + 1) / 2.0)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    else:   # 只有一个人的情况
        if self_loop:
            edge_index = [[0], [0]]
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            edge_index = edge_index.contiguous()
            edge_weight = torch.tensor([1], dtype=torch.float32)
        else:
            edge_index = [[], []]
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            edge_index = edge_index.contiguous()
            edge_weight = torch.tensor([], dtype=torch.float32)
    return edge_index.cuda(), edge_weight.cuda()

def generate_graph_index(rel_num, self_loop):
    numNode = rel_num
    edge_index, edge_weight = [], []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if self_loop:
                    edge_index.append([i, j])
                else:
                    if i != j:
                        edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    else:   # 只有一个人的情况
        edge_index = [[], []]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.contiguous()
    return edge_index.cuda()

def create_edge_index_and_weight(node_features, self_loop=False):
    """
    Calculates edge_index and edge_weight for a fully connected graph with optional self-loops.

    Args:
        node_features: A PyTorch tensor of shape (N, d) representing node features.
        self_loop: A boolean indicating whether to include self-loops in the graph.

    Returns:
        edge_index: A PyTorch tensor of shape (2, num_edges) representing the edges.
        edge_weight: A PyTorch tensor of shape (num_edges,) representing the edge weights.
    """
    N, d = node_features.shape
    if N < 1:
        return None, None

    # Efficiently create edge_index for a fully connected graph (with optional self-loops)
    if self_loop:
        row, col = torch.meshgrid(torch.arange(N), torch.arange(N))
        edge_index = torch.stack([row.flatten(), col.flatten()])
    else:
        row, col = torch.meshgrid(torch.arange(N), torch.arange(N))
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]])

    # Normalize node features
    node_features_norm = F.normalize(node_features, p=2, dim=1)

    # Calculate cosine similarity
    dot_product = torch.matmul(node_features_norm, node_features_norm.t())
    if self_loop:
        cosine_similarity = dot_product
    else:
        cosine_similarity = dot_product[mask]

    # Normalize cosine similarity to the range 0-1
    normalized_cosine_similarity = (cosine_similarity + 1) / 2

    # Convert cosine similarity to edge weights
    edge_weight = normalized_cosine_similarity.view(-1)

    return edge_index, edge_weight

def create_edge_index(N, self_loop=False):
    """
    Calculates edge_index for a fully connected graph with optional self-loops.

    Args:
        N: Number of nodes.
        self_loop: A boolean indicating whether to include self-loops in the graph.

    Returns:
        edge_index: A PyTorch tensor of shape (2, num_edges) representing the edges.
    """
    if N < 1:
        return None, None

    # Efficiently create edge_index for a fully connected graph (with optional self-loops)
    if self_loop:
        row, col = torch.meshgrid(torch.arange(N), torch.arange(N))
        edge_index = torch.stack([row.flatten(), col.flatten()])
    else:
        row, col = torch.meshgrid(torch.arange(N), torch.arange(N))
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]])

    return edge_index

def cos_sim(v1, v2):
    norm_v1 = torch.norm(v1, p=2)
    norm_v2 = torch.norm(v2, p=2)

    # 使用 epsilon 避免除以零
    epsilon = 1e-8
    norm_v1 = torch.where(norm_v1 == 0, torch.tensor(epsilon).to(v1.device), norm_v1)
    norm_v2 = torch.where(norm_v2 == 0, torch.tensor(epsilon).to(v2.device), norm_v2)

    v1_normalized = v1 / norm_v1
    v2_normalized = v2 / norm_v2

    return v1_normalized @ v2_normalized.t()