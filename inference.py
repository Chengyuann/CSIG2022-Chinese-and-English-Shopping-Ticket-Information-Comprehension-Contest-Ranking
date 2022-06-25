from cgitb import text
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchtext.data import Field, RawField
import pickle
import numpy as np

import cv2

from model.PICK.model import pick as pick_arch_model
from model.PICK.data_utils.documents import Document,normalize_relation_features, sort_box_with_list
from model.PICK.utils.util import iob_index_to_str,text_index_to_str
from model.PICK.utils.class_utils import keys_vocab_cls


from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans



class BaselineService:

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        # 调用自定义函数加载模型

        if False and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        self.max_box_num = 300
        self.max_transcript_len = 50
        self.image_resize = (480, 960)
        self.entities = ["company", "date", "total", "tax", "name", "cnt", "price"]

        self.textSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
        self.textSegmentsField.vocab = keys_vocab_cls

        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.pick_model = config.init_obj('model_arch', pick_arch_model)
        self.pick_model = self.pick_model.to(self.device)
        self.pick_model.load_state_dict(state_dict)
        self.pick_model.eval()

    def _preprocess(self, request_data):
        data = None
        for k, v in request_data.items():
            for file_name, file_content in v.items():
                data = pickle.load(file_content)

        image = data['image']
        ocr = data['ocr']
        id = data['id']
        boxes_and_transcripts_data = [(i, box['points'], box['label']) for i, box in enumerate(ocr)]
        boxes_and_transcripts_data = sort_box_with_list(boxes_and_transcripts_data)
        boxes = []
        transcripts = []

        for _, points, transcript in boxes_and_transcripts_data:
            if len(transcript) == 0 :
                transcript = ' '
            mp=[]
            for p in points:
                mp+=p
            boxes.append(mp)
            transcripts.append(transcript)

        boxes_num = min(len(boxes), self.max_box_num)
        transcripts_len = min(max([len(t) for t in transcripts[:boxes_num]]), self.max_transcript_len)
        mask = np.zeros((boxes_num, transcripts_len), dtype=int)

        relation_features = np.zeros((boxes_num, boxes_num, 6))

        height, width, _ = image.shape

        image = cv2.resize(image, self.image_resize, interpolation=cv2.INTER_LINEAR)
        x_scale = self.image_resize[0] / width
        y_scale = self.image_resize[1] / height

        min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in boxes[:boxes_num]]

        resized_boxes = []
        for i in range(boxes_num):
            box_i = boxes[i]
            transcript_i = transcripts[i]
            resized_box_i = [int(np.round(pos * x_scale)) if i%2 == 0 else int(np.round(pos * y_scale)) for i, pos in enumerate(box_i)]
            resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
            resized_box_i = cv2.boxPoints(resized_rect_output_i)
            resized_box_i = resized_box_i.reshape((8,))
            resized_boxes.append(resized_box_i)
            Document.relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                                        transcripts)

        relation_features = normalize_relation_features(relation_features, width=width, height=height)
        text_segments = [list(trans) for trans in transcripts[:boxes_num]]
        texts, texts_len = self.textSegmentsField.process(text_segments)
        texts = texts[:, :transcripts_len].cpu().numpy()
        texts_len = np.clip(texts_len.cpu().numpy(), 0, transcripts_len)
        text_segments = ( texts, texts_len)

        for i in range(boxes_num):
            mask[i, :texts_len[i]] = 1

        whole_image = RawField().preprocess(image)
        boxes_coordinate = RawField().preprocess(resized_boxes)
        relation_features = RawField().preprocess(relation_features)
        mask = RawField().preprocess(mask)
        boxes_num = RawField().preprocess(boxes_num)
        transcripts_len = RawField().preprocess(transcripts_len)

        return dict(
            whole_image=torch.stack([self.trsfm(whole_image)], dim=0).float().to(self.device),
            relation_features=torch.stack([torch.FloatTensor(relation_features)], dim=0).to(self.device),
            boxes_coordinate=torch.stack([torch.FloatTensor(boxes_coordinate)], dim=0).to(self.device),
            text_segments=torch.stack([torch.LongTensor(text_segments[0])], dim=0).to(self.device),
            text_length=torch.stack([torch.LongTensor(text_segments[1])], dim=0).to(self.device),
            mask=torch.stack([torch.ByteTensor(mask)], dim=0).to(self.device),
            file_id=[id],
            transcripts=transcripts
        )

    def _postprocess(self, data):
        pass
        return data
    
    def _inference(self, input_data):
        output = self.pick_model(**input_data)
        logits = output['logits']  # (B, N*T, out_dim)
        new_mask = output['new_mask']
        image_indexs = input_data['file_id']  # (B,)
        text_segments = input_data['text_segments']  # (B, num_boxes, T)
        mask = input_data['mask']
        # List[(List[int], torch.Tensor)]
        best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
        predicted_tags = []
        for path, score in best_paths:
            predicted_tags.append(path)

        # convert iob index to iob string
        decoded_tags_list = iob_index_to_str(predicted_tags)
        # union text as a sequence and convert index to string
        decoded_texts_list = text_index_to_str(text_segments, mask)
        
        for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
            # List[ Tuple[str, Tuple[int, int]] ]
            spans = bio_tags_to_spans(decoded_tags, [])
            spans = sorted(spans, key=lambda x: x[1][0])

            entities = []  # exists one to many case
            for entity_name, range_tuple in spans:
                entity = dict(entity_name=entity_name,
                              text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                entities.append(entity)

            result = dict(
                company="",
                date="",
                total="",
                tax="",
                items=[]
            )
            new_item = dict()
            for e in entities:
                entity_name=e['entity_name']
                text=e['text']
                if entity_name in result:
                    result[entity_name] = text
                elif entity_name in ["price", "cnt", "name"]:
                    if entity_name in new_item:
                        if "cnt" not in new_item:
                            new_item['cnt'] = "1"
                        result['items'].append(new_item)
                        new_item = dict()
                    new_item[entity_name] = text
                    
            return result

###############################################################################
# 请根据实际路径修改下面两个变量
model_path="saved/models/PICK_Default/test_0428_161241/model_best.pth"
data_path="./test_data.pkl"
###############################################################################

###############################################################################
# 下方代码毋需改动
infer_api=BaselineService("", model_path)

file_content=open(data_path,"rb")
request_data={
    "file1":{
        "test_data.pkl": file_content
    }
}

input_data=infer_api._preprocess(request_data)
result=infer_api._inference(input_data)
final_result=infer_api._postprocess(result)
file_content.close()
print(final_result)