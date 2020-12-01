# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import torch
import numpy as np

from torch import Tensor
from torch.nn import functional as F

# Based on the huggingface beam search
# https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py


def _generate_beam_search(
    self,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    early_stopping,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    pad_token_id,
    eos_token_id,
    batch_size,
    num_return_sequences,
    length_penalty,
    num_beams,
    vocab_size,
    encoder_outputs,
    attention_mask,
    use_cache,
    model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """
    assert not do_sample
    num_groups = 3
    assert num_beams % num_groups == 0
    group_size = num_beams // num_groups
    diversity_strength = 2
    #print("DIVERSE BEAM SEARCH", diversity_strength)
    #tokenizer = get_tokenizer()

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, num_groups, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    #if do_sample is False:
    #    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if self._use_cache(outputs, use_cache):
            past = outputs[1]
        if self.config.is_encoder_decoder and do_sample is False:
            # TODO (PVP) still a bit hacky here - there might be a better solution
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)        
        scores = self.postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        # diversity penalty buffer
        diversity_buf = torch.zeros(scores[0, :].size()).to(scores)

        next_scores = []
        next_tokens = []
        # diverse beam search
        for group in range(num_groups):
            #print("GROUP", group)
            start_g, end_g = group*group_size, (group+1)*group_size
            scores_g = scores[start_g : end_g, :]
            beam_scores_g = beam_scores[:, None].expand_as(scores)[start_g : end_g, :]
            next_scores_g = scores_g + diversity_buf + beam_scores_g  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores_g = next_scores_g.view(
                batch_size, group_size * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores_g, next_tokens_g = torch.topk(next_scores_g, 2*group_size, dim=1, largest=True, sorted=True)
            #print(next_scores_g)
            #print(next_tokens_g)

            diversity_buf[next_tokens_g[0] % vocab_size] += -diversity_strength
            next_scores.append(next_scores_g.clone())
            next_tokens.append(next_tokens_g.clone())

        next_scores = torch.stack(next_scores, dim=1).view(batch_size, 2*num_beams)
        next_tokens = torch.stack(next_tokens, dim=1).view(batch_size, 2*num_beams)
        #print(next_tokens)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2*num_beams)
        # next batch beam content
        next_batch_beam = []

        #print()
        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            for group in range(num_groups):
                next_sent_beam_group = []
                start_g, end_g = 2*group*group_size, 2*(group+1)*group_size
                #print("GROUP", group)
                #tt = next_tokens[batch_idx, start_g:end_g].tolist()
                #print(tt)
                #print(tokenizer.decode(np.array(tt)%50257))


                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx, start_g:end_g], next_scores[batch_idx, start_g:end_g])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx*num_beams + group*group_size + beam_id
                    # add to generated hypotheses if end of sentence
                    if ((eos_token_id is not None) and (token_id.item() == eos_token_id)) or token_id.item() == 41757: #new bash
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(), group
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam_group.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam_group) == group_size:
                        #print(next_sent_beam)
                        break
                next_sent_beam += next_sent_beam_group

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )


            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"



        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        """
        print(beam_idx)
        print("PREV INPUT IDS")
        for i in input_ids:
            print(tokenizer.decode(i))
        """

        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1
        """
        print()
        print("NEXT INPUT IDS")
        for i in input_ids:
            print(tokenizer.decode(i))
        """


        # re-order internal states
        if past is not None:
            past = self._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        """
        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )
        """

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score, beam_id//group_size)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted([x for y in hypotheses.beams for x in y], key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

    return decoded

class BeamHypotheses(object):
    def __init__(self, num_beams, num_groups, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.group_size = num_beams // num_groups
        self.beams = [[] for _ in range(num_groups)]
        self.worst_score = [1e9 for _ in range(num_groups)]


    def add(self, hyp, sum_logprobs, group):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self.beams[group]) < self.group_size or score > self.worst_score[group]:
            self.beams[group].append((score, hyp))
            if len(self.beams[group]) > self.group_size:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams[group])])
                del self.beams[group][sorted_scores[0][1]]
                self.worst_score[group] = sorted_scores[1][0]
            else:
                self.worst_score[group] = min(score, self.worst_score[group])


    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if any(len(group) < self.group_size for group in self.beams):
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            worst = min(self.worst_score)
            ret = worst >= cur_score
            return ret