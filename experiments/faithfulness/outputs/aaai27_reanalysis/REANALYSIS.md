# AAAI-27 re-analysis (no new inference)

## A1. Held-out faithfulness utility (top-rank vs non-top deletion)

Positive `mean_top_minus_non_top` means deleting the top-|SV| segment
reduces response similarity more than the average non-top segment.
TF-IDF is the utility used inside the Shapley game (circularity caveat);
embedding and sequence-match are held-out utilities.

| condition | speech | metric | n_samples | n_deletions | mean_top_drop | mean_non_top_drop | mean_top_minus_non_top | paired_t | paired_p_one_sided | cohen_dz | spearman_abs_sv_vs_drop | mean_within_sample_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SM2S-1k (male TTS) | TTS | embedding | 94 | 658 | 0.4997 | 0.4945 | -0.0049 | -0.2079 | 0.5821 | -0.0214 | None | None |
| SM2S-1k (male TTS) | TTS | tfidf | 94 | 658 | 0.3209 | 0.3446 | -0.0478 | -2.1228 | 0.9818 | -0.2190 | None | None |
| SM2S-1k (male TTS) | TTS | seqmatch | 94 | 658 | 0.2964 | 0.3013 | -0.0227 | -1.0685 | 0.8560 | -0.1102 | None | None |
| SM2F-1k (female TTS) | TTS | embedding | 100 | 708 | 0.5018 | 0.4957 | -0.0020 | -0.0987 | 0.5392 | -0.0099 | None | None |
| SM2F-1k (female TTS) | TTS | tfidf | 100 | 708 | 0.3001 | 0.3348 | -0.0616 | -2.8326 | 0.9972 | -0.2833 | None | None |
| SM2F-1k (female TTS) | TTS | seqmatch | 100 | 708 | 0.2657 | 0.2950 | -0.0482 | -2.4506 | 0.9920 | -0.2451 | None | None |
| SO2S-1k (orig TTS) | TTS | embedding | 99 | 703 | 0.5449 | 0.5032 | 0.0328 | 1.4599 | 0.0738 | 0.1467 | None | None |
| SO2S-1k (orig TTS) | TTS | tfidf | 99 | 703 | 0.4115 | 0.3753 | 0.0172 | 0.6513 | 0.2582 | 0.0655 | None | None |
| SO2S-1k (orig TTS) | TTS | seqmatch | 99 | 703 | 0.3620 | 0.3351 | 0.0112 | 0.4891 | 0.3129 | 0.0492 | None | None |
| SO2S-500 (LibriSpeech) | natural | embedding | 100 | 468 | 0.6288 | 0.5762 | 0.0514 | 2.1280 | 0.0179 | 0.2128 | None | None |
| SO2S-500 (LibriSpeech) | natural | tfidf | 100 | 468 | 0.5931 | 0.4820 | 0.1075 | 2.9895 | 0.0018 | 0.2989 | None | None |
| SO2S-500 (LibriSpeech) | natural | seqmatch | 100 | 468 | 0.5124 | 0.4406 | 0.0682 | 2.4222 | 0.0086 | 0.2422 | None | None |

## A2. AOPC (area over cumulative top-|SV| removal curve, x = k/n)

| condition | speech | metric | n_samples | aopc_sv_order | aopc_single_mean | mean_final_cumulative_drop |
| --- | --- | --- | --- | --- | --- | --- |
| SM2S-1k (male TTS) | TTS | embedding | 94 | 0.5808 | 0.4953 | 0.7461 |
| SM2S-1k (male TTS) | TTS | tfidf | 94 | 0.6644 | 0.3412 | 0.9338 |
| SM2S-1k (male TTS) | TTS | seqmatch | 94 | 0.5561 | 0.3006 | 0.7389 |
| SM2F-1k (female TTS) | TTS | embedding | 100 | 0.6015 | 0.4965 | 0.7318 |
| SM2F-1k (female TTS) | TTS | tfidf | 100 | 0.6518 | 0.3299 | 0.9358 |
| SM2F-1k (female TTS) | TTS | seqmatch | 100 | 0.5534 | 0.2909 | 0.7419 |
| SO2S-1k (orig TTS) | TTS | embedding | 99 | 0.6134 | 0.5091 | 0.7441 |
| SO2S-1k (orig TTS) | TTS | tfidf | 99 | 0.7273 | 0.3804 | 0.9359 |
| SO2S-1k (orig TTS) | TTS | seqmatch | 99 | 0.5947 | 0.3389 | 0.7379 |
| SO2S-500 (LibriSpeech) | natural | embedding | 100 | 0.6166 | 0.5875 | 0.7956 |
| SO2S-500 (LibriSpeech) | natural | tfidf | 100 | 0.7084 | 0.5057 | 0.8884 |
| SO2S-500 (LibriSpeech) | natural | seqmatch | 100 | 0.6000 | 0.4560 | 0.7183 |

## A2. Attribution-flatness / saturation diagnostic

| condition | speech | n_samples | flat_abs_sv_sample_share | mean_abs_sv_entropy_norm | saturation_frac_drop_ge_0_8 |
| --- | --- | --- | --- | --- | --- |
| SM2S-1k (male TTS) | TTS | 94 | 1.0000 | 1.0000 | 0.1185 |
| SM2F-1k (female TTS) | TTS | 100 | 1.0000 | 1.0000 | 0.1116 |
| SO2S-1k (orig TTS) | TTS | 99 | 1.0000 | 1.0000 | 0.1195 |
| SO2S-500 (LibriSpeech) | natural | 100 | 1.0000 | 1.0000 | 0.2265 |
