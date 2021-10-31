#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
import editdistance
import torch
from torch.nn import CTCLoss
import string
from transformers import pipeline
import re
import random          


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        self.optimizer.zero_grad()
        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        self.backward_loss(loss)
        self.optimizer.step()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")

        x = self.models["encoder"](x)
        global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def check_spell(self, str_x ):
        str_spell = str_x.copy()
        #str_yaspell = str_x.copy()
        str_masked = str_x.copy()
        str_masked_T5 = str_x.copy()
        exclude = set(string.punctuation)

        for x in str_spell:
            x_lst = x.split()
            masked_lst = x_lst.copy()
            str_corrected = x[:]
            misspelled = []
            x_masked = x[:]
            x_masked_T5 = x[:]
            stripped = x[:]
            stripped = ''.join(ch for ch in stripped if ch not in exclude)
            stripped_lst = stripped.split()

            #fixed = self.yaspeller.spelled(stripped)

            misspelled = self.pyspell.unknown(stripped_lst)
            misspelled = list(misspelled)
            if len(misspelled) > 0:
                if len(x_lst) > 1:
                    pos = random.randint(1, len(misspelled))
                    worst = misspelled[pos] #worst = max(misspelled, key=len) #longest word masked
                    x_masked = re.sub(r"\b%s\b" % worst, '[MASK]', x_masked, 1)
                    str_masked = [w.replace(x, x_masked) for w in str_masked]
                    
                    for i in range(len(misspelled)):                        
                        worst = misspelled[i]                       
                        x_masked_T5 = re.sub(r"\b%s\b" % worst, '<extra_id_' + str(i) + '>', x_masked, 1)
                        str_masked_T5 = [w.replace(x, x_masked_T5) for w in str_masked_T5]                        

                for pred in misspelled:
                    if ( not(pred.isdigit()) and len(pred) > 2):
                        corrected = self.pyspell.correction(pred) 
                        x_lst = [w.replace(pred, corrected) for w in x_lst]
            
            str_corrected = " ".join(x_lst)
            str_spell = [w.replace(x, str_corrected) for w in str_spell]
            #str_yaspell = [w.replace(x, fixed) for w in str_yaspell]

        return str_spell, str_masked, str_masked_T5
             
    def improve_LM(self, str_x, model, tokenizer, model_name):
        str_improved = str_x.copy()
        unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
        if model_name == 'ruRoberta':
            str_improved = [w.replace('[MASK]', '<mask>') for w in str_improved]
        for x in str_improved:
            if (x.find('[MASK]') != -1 or x.find('<mask>') != -1):
                x_improved = [i['sequence'] for i in unmasker(x, top_k=1)]
            else:
                x_improved = [x]
            str_improved = [w.replace(x, x_improved[0]) for w in str_improved]
        return str_improved     
        
    def trimT5(selfself, x, x_improved, count):
        for i in range(count):
            start_token = '<extra_id_'+ str(i) +'>'
            _0_index = x.index(start_token)
            _result_prefix = x[:_0_index]
            _result_suffix = x[_0_index + 12:]
            end_token = '<extra_id_'+ str(i+1) +'>'
            if start_token in x_improved:
                _start_token_index = x_improved.index(start_token)+12
            else:
                _start_token_index = 12

            if end_token in x_improved:
                _end_token_index = x_improved.index(end_token)
                x = _result_prefix + (x_improved[_start_token_index:_end_token_index]).strip() + _result_suffix
            elif '</s>' in x_improved:
                _end_token_index = x_improved.index('</s>')
                x = _result_prefix + (x_improved[_start_token_index:_end_token_index]).strip() + _result_suffix
            else:
                x = _result_prefix + (x_improved[_start_token_index:]).strip() + _result_suffix
        return x  
        
    def improve_T5(self, str_x, model, tokenizer):
        str_improved = str_x.copy()
        str_improved = [w.replace('[MASK]', '<extra_id_0>') for w in str_improved]
        for x in str_improved:
            if (x.find('<extra_id_0>') != -1):
                input_ids = tokenizer(x, return_tensors='pt').input_ids
                out_ids = model.generate(input_ids=input_ids, max_length=10, eos_token_id=tokenizer.eos_token_id,
                                         early_stopping=True)
                x_improved = tokenizer.decode(out_ids[0][1:])
                x_improved = [self.trimT5(x, x_improved)]
            else:
                x_improved = [x]
            str_improved = [w.replace(x, x_improved[0]) for w in str_improved]
        return str_improved

    def improve_T5_multiple(self, str_x, model, tokenizer):
        str_improved = str_x.copy()
        for x in str_improved:
            count =  x.count('<extra_id_')
            if (x.find('<extra_id_0>') != -1):
                input_ids = tokenizer(x, return_tensors='pt').input_ids
                out_ids = model.generate(input_ids=input_ids, max_length=10, eos_token_id=tokenizer.eos_token_id,
                                         early_stopping=True)
                x_improved = tokenizer.decode(out_ids[0][1:])
                x_improved = self.trimT5(x, x_improved, count)
                x_improved = [re.sub('<extra_id_\d+>', '', x_improved).replace('>', '')]
            else:
                x_improved = [x]
            str_improved = [w.replace(x, x_improved[0]) for w in str_improved]
        return str_improved
                             
    def compute_metrics(self, x, y, x_len, y_len, loss=None, metric_names=list()):
        batch_size = y.shape[0]
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]
        
        str_spell, str_masked, str_masked_T5 = self.check_spell(str_x)          
        str_ruBert_base = self.improve_LM(str_masked, self.model_ruBert_base, self.tokenizer_ruBert_base, 'ruBert')
        #print("str_ruBert_base", str_ruBert_base)
        str_ruRoberta = self.improve_LM(str_masked, self.model_ruRoberta, self.tokenizer_ruRoberta, 'ruRoberta')
        #print("str_ruRoberta", str_ruRoberta)
        str_ruBert_large = self.improve_LM(str_masked, self.model_ruBert_large, self.tokenizer_ruBert_large, 'ruBert')
        #print("str_ruBert_large", str_ruBert_large)

        str_ruT5_base = self.improve_T5_multiple(str_masked_T5, self.model_ruT5_base, self.tokenizer_ruT5_base)
        #print("str_ruT5_base", str_ruT5_base)
        str_ruT5_large = self.improve_T5_multiple(str_masked_T5, self.model_ruT5_large, self.tokenizer_ruT5_large)
        #print("str_ruT5_large", str_ruT5_large)
                     
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["cer_spell"] = [editdistance.eval(u, v) for u,v in zip(str_y, str_spell)]
                metrics["cer_ruT5_base"] = [editdistance.eval(u, v) for u, v in zip(str_y, str_ruT5_base)]
                metrics["cer_ruT5_large"] = [editdistance.eval(u, v) for u, v in zip(str_y, str_ruT5_large)]
                metrics["cer_ruBert_base"] = [editdistance.eval(u, v) for u,v in zip(str_y, str_ruBert_base)]
                metrics["cer_ruBert_large"] = [editdistance.eval(u, v) for u,v in zip(str_y, str_ruBert_large)]
                metrics["cer_ruRoberta"] = [editdistance.eval(u, v) for u,v in zip(str_y, str_ruRoberta)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
                metrics["wer_spell"] = edit_wer_from_list(str_y, str_spell)
                metrics["wer_ruT5_base"] = edit_wer_from_list(str_y, str_ruT5_base)
                metrics["wer_ruT5_large"] = edit_wer_from_list(str_y, str_ruT5_large)
                metrics["wer_ruBert_base"] = edit_wer_from_list(str_y, str_ruBert_base)
                metrics["wer_ruBert_large"] = edit_wer_from_list(str_y, str_ruBert_large)
                metrics["wer_ruRoberta"] = edit_wer_from_list(str_y, str_ruRoberta)                                                                         
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
            elif metric_name == "pred_spell":
                metrics["pred_spell"] = [str_spell, ]
            elif metric_name == "pred_ruT5_base":
                metrics["pred_ruT5_base"] = [str_ruT5_base, ]
            elif metric_name == "pred_ruT5_large":
                metrics["pred_ruT5_large"] = [str_ruT5_large, ]
            elif metric_name == "pred_ruBert_base":
                metrics["pred_ruBert_base"] = [str_ruBert_base, ]
            elif metric_name == "pred_ruBert_large":
                metrics["pred_ruBert_large"] = [str_ruBert_large, ]
            elif metric_name == "pred_ruRoberta":
                metrics["pred_ruRoberta"] = [str_ruRoberta, ]                                                                                                                                                
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss / metrics["nb_chars"]
        metrics["nb_samples"] = len(x)
        return metrics
