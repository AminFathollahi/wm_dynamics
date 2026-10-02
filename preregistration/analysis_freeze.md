# Analysis freeze: cell tables

Generated from `analysis_freeze.json`. Status is frozen or not_askable.

## Link 1: stimulation to firing

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H5 | alagapan_phase_stimulation | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H5 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ds005034 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H5 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | haslacher_clam_tacs | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H5 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus; the delay-period optogenetic arm lies outside the stimulation definition |
| H5 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | macaque_pfc_microstimulation | frozen | Y | this link has no behavioural outcome | category_selective | none used | session | 11 | 0.758/ | never_pooled |
| H5 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ram_ds005489_openloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H5 | ram_ds005557_closedloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H5 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 2: stimulation to population state

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H5 | alagapan_phase_stimulation | frozen | P | this link has no behavioural outcome | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | never_pooled |
| H5 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ds005034 | frozen | P | this link has no behavioural outcome | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 25 | 0.535/ | never_pooled |
| H5 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | haslacher_clam_tacs | frozen | Y | this link has no behavioural outcome | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | never_pooled |
| H5 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus; the delay-period optogenetic arm lies outside the stimulation definition |
| H5 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | macaque_pfc_microstimulation | frozen | Y | this link has no behavioural outcome | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | never_pooled |
| H5 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | ram_ds005489_openloop | frozen | Y | this link has no behavioural outcome | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | never_pooled |
| H5 | ram_ds005557_closedloop | frozen | Y | this link has no behavioural outcome | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | never_pooled |
| H5 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H5 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 3: stimulation to behaviour

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H6 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | none used | patient | 3 | /0.159 | never_pooled |
| H6 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ds005034 | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H6 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | none used | patient | 46 | 0.403/0.021 | never_pooled |
| H6 | inagaki_alm5 | not_askable | N | | | | | | | behaviour absent for this preparation (no comparable per-trial accuracy family) |
| H6 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | macaque_pfc_microstimulation | frozen | Y | log reaction time (correct trials only) | all_units | none used | session | 11 | 0.758/ | never_pooled |
| H6 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ram_ds005489_openloop | frozen | Y | recall of the word (0/1) | all_contacts_of_the_region_all_bands | none used | patient | 37 | 0.447/ | never_pooled |
| H6 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | none used | patient | 16 | 0.651/ | never_pooled |
| H6 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 4: firing to population state

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | alagapan_phase_stimulation | not_askable | N | | | | | | | the only label is list length (present in the raw behaviour file; not read by the loader), which is a task condition and not a per-trial content label |
| H1 | dandi_000004 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 59 | 0.358/0.030 | H1|human|units|new_old_recognition|link4 |
| H1 | dandi_000469 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 20 | 0.591/0.098 | H1|human|units|sternberg_item_recognition|link4 |
| H1 | dandi_000574 | not_askable | P | | | | | | | the only label is set size, 4/6/8; no per-trial item identity is released, which is a task condition and not a per-trial content label |
| H1 | dandi_000673 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 5 | 0.963/0.109 | H1|human|units|sternberg_item_recognition|link4 |
| H1 | dandi_001187 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 39 | 0.436/0.036 | H1|human|units|sternberg_item_recognition|link4 |
| H1 | ds004752 | not_askable | N | | | | | | | the only label is set size, 4/6/8, which is a task condition and not a per-trial content label |
| H1 | ds005034 | not_askable | N | | | | | | | the only label is operation, 3 classes, which is a task condition and not a per-trial content label |
| H1 | ds006848 | not_askable | N | | | | | | | the only label is presentation mode, 4 classes, which is a task condition and not a per-trial content label |
| H1 | haslacher_clam_tacs | not_askable | N | | | | | | | the only label is None, which is a task condition and not a per-trial content label |
| H1 | inagaki_alm5 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | session | 23 | 0.556/0.029 | H1|mouse|units|instructed_lick_delayed_response|link4 |
| H1 | kai_miller_nback | not_askable | N | | | | | | | the only label is n-back load, 0/1/2, which is a task condition and not a per-trial content label |
| H1 | macaque_pfc_microstimulation | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | H1|macaque|units|spatial_delayed_saccade|link4 |
| H1 | panichello_2024 | frozen | P | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | session | 25 | 0.535/ | H1|macaque|units|spatial_delayed_saccade|link4 |
| H1 | pfc3 | frozen | P | this link has no behavioural outcome | all_recorded_units_of_the_region_set (as published) | units: demixed_principal_components, gaussian_process_factor_analysis | animal | 4 | 0.993/ | H1|macaque|units|spatial_delayed_match_to_sample|link4 |
| H1 | pfc4 | not_askable | P | | | | | | | the population state needs at least 8 units in the region-session and this corpus records at most 7 units per session |
| H1 | ram_ds005489_openloop | not_askable | N | | | | | | | the only label is None, which is a task condition and not a per-trial content label |
| H1 | ram_ds005557_closedloop | not_askable | N | | | | | | | the only label is None, which is a task condition and not a per-trial content label |
| H1 | watters_2026 | frozen | Y | this link has no behavioural outcome | category_selective (F: selective-unit rate and tuning contrast); all_units (P) | units: demixed_principal_components, gaussian_process_factor_analysis | session | 41 | 0.426/ | H1|macaque|units|spatial_delayed_saccade|link4 |
| H1 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H1 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |

## Link 5: firing to behaviour

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H3 | alagapan_phase_stimulation | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | dandi_000004 | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | patient | 59 | 0.358/0.030 | H3|human|units|new_old_recognition|link5 |
| H3 | dandi_000469 | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | patient | 20 | 0.591/0.098 | H3|human|units|sternberg_item_recognition|link5 |
| H3 | dandi_000574 | frozen | P | log reaction time (correct trials only) | all_units (population rate; no tuning contrast is defined without a per-trial content label) | none used | patient | 9 | 0.816/0.066 | H3|human|units|sternberg_item_recognition|link5 |
| H3 | dandi_000673 | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | patient | 5 | 0.963/0.109 | H3|human|units|sternberg_item_recognition|link5 |
| H3 | dandi_001187 | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | patient | 39 | 0.436/0.036 | H3|human|units|sternberg_item_recognition|link5 |
| H3 | ds004752 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ds005034 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ds006848 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | haslacher_clam_tacs | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | inagaki_alm5 | not_askable | N | | | | | | | behaviour absent for this preparation (no comparable per-trial accuracy family) |
| H3 | kai_miller_nback | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | session | 11 | 0.758/ | H3|macaque|units|spatial_delayed_saccade|link5 |
| H3 | panichello_2024 | frozen | P | accuracy (correct and error trials) | category_selective (tuning contrast); all_units (population rate) | none used | session | 25 | 0.535/ | H3|macaque|units|spatial_delayed_saccade|link5 |
| H3 | pfc3 | not_askable | N | | | | | | | no trial-level correctness field was found in the released trial structure |
| H3 | pfc4 | frozen | P | accuracy (correct and error trials) | category_selective (tuning contrast); all_units (population rate) | none used | session | 196 | 0.199/0.026 | H3|macaque|units|vibrotactile_delayed_discrimination|link5 |
| H3 | ram_ds005489_openloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ram_ds005557_closedloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | watters_2026 | frozen | P | log reaction time (correct trials only) | category_selective (tuning contrast); all_units (population rate) | none used | session | 41 | 0.426/ | H3|macaque|units|spatial_delayed_saccade|link5 |
| H3 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |

## Link 6: population state to behaviour

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H2 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | H2|human|intracranial_field_potentials|sternberg_item_recognition|link6 |
| H2 | dandi_000004 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 59 | 0.358/0.030 | H2|human|units|new_old_recognition|link6 |
| H2 | dandi_000469 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 20 | 0.591/0.098 | H2|human|units|sternberg_item_recognition|link6 |
| H2 | dandi_000574 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis; depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.066 | H2|human|units|sternberg_item_recognition|link6 |
| H2 | dandi_000673 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 5 | 0.963/0.109 | H2|human|units|sternberg_item_recognition|link6 |
| H2 | dandi_001187 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 39 | 0.436/0.036 | H2|human|units|sternberg_item_recognition|link6 |
| H2 | ds004752 | frozen | Y | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 15 | 0.669/0.048 | H2|human|depth_field_potentials|sternberg_item_recognition|link6 |
| H2 | ds005034 | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H2 | ds006848 | frozen | P | graded recall score (digits correct in order, 0 to 7) (replaces accuracy) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 30 | 0.492/0.036 | H2|human|scalp_eeg|serial_digit_recall|link6 |
| H2 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | H2|human|scalp_eeg|visual_retention_with_phase_locked_stimulation|link6 |
| H2 | inagaki_alm5 | not_askable | N | | | | | | | behaviour absent for this preparation (no comparable per-trial accuracy family) |
| H2 | kai_miller_nback | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H2 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | H2|macaque|units|spatial_delayed_saccade|link6 |
| H2 | panichello_2024 | frozen | Y | accuracy (correct and error trials) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 25 | 0.535/ | H2|macaque|units|spatial_delayed_saccade|link6 |
| H2 | pfc3 | not_askable | N | | | | | | | no trial-level correctness field was found in the released trial structure |
| H2 | pfc4 | not_askable | P | | | | | | | the population state needs at least 8 units in the region-session and this corpus records at most 7 units per session |
| H2 | ram_ds005489_openloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | H2|human|depth_field_potentials|delayed_free_recall|link6 |
| H2 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | H2|human|depth_field_potentials|delayed_free_recall|link6 |
| H2 | watters_2026 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 41 | 0.426/ | H2|macaque|units|spatial_delayed_saccade|link6 |
| H2 | wolff_eeg_impulse experiment_1 | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 30 | 0.492/0.014 | H2|human|scalp_eeg|retro_cue_orientation_report|link6|experiment_1 |
| H2 | wolff_eeg_impulse experiment_2 | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 18 | 0.619/ | H2|human|scalp_eeg|retro_cue_orientation_report|link6|experiment_2 |
| H4 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | H4|human|intracranial_field_potentials|sternberg_item_recognition|link6 |
| H4 | dandi_000004 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 59 | 0.358/0.030 | H4|human|units|new_old_recognition|link6 |
| H4 | dandi_000469 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 20 | 0.591/0.098 | H4|human|units|sternberg_item_recognition|link6 |
| H4 | dandi_000574 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis; depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.066 | H4|human|units|sternberg_item_recognition|link6 |
| H4 | dandi_000673 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 5 | 0.963/0.109 | H4|human|units|sternberg_item_recognition|link6 |
| H4 | dandi_001187 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 39 | 0.436/0.036 | H4|human|units|sternberg_item_recognition|link6 |
| H4 | ds004752 | frozen | Y | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 15 | 0.669/0.048 | H4|human|depth_field_potentials|sternberg_item_recognition|link6 |
| H4 | ds005034 | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H4 | ds006848 | frozen | P | graded recall score (digits correct in order, 0 to 7) (replaces accuracy) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 30 | 0.492/0.036 | H4|human|scalp_eeg|serial_digit_recall|link6 |
| H4 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | H4|human|scalp_eeg|visual_retention_with_phase_locked_stimulation|link6 |
| H4 | inagaki_alm5 | not_askable | N | | | | | | | behaviour absent for this preparation (no comparable per-trial accuracy family) |
| H4 | kai_miller_nback | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H4 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | H4|macaque|units|spatial_delayed_saccade|link6 |
| H4 | panichello_2024 | frozen | Y | accuracy (correct and error trials) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 25 | 0.535/ | H4|macaque|units|spatial_delayed_saccade|link6 |
| H4 | pfc3 | not_askable | N | | | | | | | no trial-level correctness field was found in the released trial structure |
| H4 | pfc4 | not_askable | P | | | | | | | the population state needs at least 8 units in the region-session and this corpus records at most 7 units per session |
| H4 | ram_ds005489_openloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | H4|human|depth_field_potentials|delayed_free_recall|link6 |
| H4 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | H4|human|depth_field_potentials|delayed_free_recall|link6 |
| H4 | watters_2026 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 41 | 0.426/ | H4|macaque|units|spatial_delayed_saccade|link6 |
| H4 | wolff_eeg_impulse experiment_1 | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 30 | 0.492/0.014 | H4|human|scalp_eeg|retro_cue_orientation_report|link6|experiment_1 |
| H4 | wolff_eeg_impulse experiment_2 | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 18 | 0.619/ | H4|human|scalp_eeg|retro_cue_orientation_report|link6|experiment_2 |

## Link 7: population state to behaviour given firing

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H3 | alagapan_phase_stimulation | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | dandi_000004 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 59 | 0.358/0.030 | H3|human|units|new_old_recognition|link7 |
| H3 | dandi_000469 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 20 | 0.591/0.098 | H3|human|units|sternberg_item_recognition|link7 |
| H3 | dandi_000574 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis; depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.066 | H3|human|units|sternberg_item_recognition|link7 |
| H3 | dandi_000673 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 5 | 0.963/0.109 | H3|human|units|sternberg_item_recognition|link7 |
| H3 | dandi_001187 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 39 | 0.436/0.036 | H3|human|units|sternberg_item_recognition|link7 |
| H3 | ds004752 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ds005034 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ds006848 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | haslacher_clam_tacs | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | inagaki_alm5 | not_askable | N | | | | | | | behaviour absent for this preparation (no comparable per-trial accuracy family) |
| H3 | kai_miller_nback | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | H3|macaque|units|spatial_delayed_saccade|link7 |
| H3 | panichello_2024 | frozen | Y | accuracy (correct and error trials) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 25 | 0.535/ | H3|macaque|units|spatial_delayed_saccade|link7 |
| H3 | pfc3 | not_askable | N | | | | | | | no trial-level correctness field was found in the released trial structure |
| H3 | pfc4 | not_askable | P | | | | | | | the population state needs at least 8 units in the region-session and this corpus records at most 7 units per session |
| H3 | ram_ds005489_openloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | ram_ds005557_closedloop | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | watters_2026 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 41 | 0.426/ | H3|macaque|units|spatial_delayed_saccade|link7 |
| H3 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |
| H3 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no single-unit or multi-unit recording, so firing rate is absent |

## Link 8: stimulation by pre-stimulation state to behaviour

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H6 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | never_pooled |
| H6 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ds005034 | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H6 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | haslacher_clam_tacs | frozen | Y | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | never_pooled |
| H6 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus; the delay-period optogenetic arm lies outside the stimulation definition |
| H6 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | never_pooled |
| H6 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | ram_ds005489_openloop | frozen | Y | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | never_pooled |
| H6 | ram_ds005557_closedloop | frozen | Y | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | never_pooled |
| H6 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H6 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 9: stimulation to population state to behaviour (mediation)

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H8 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | never_pooled |
| H8 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | ds005034 | not_askable | N | | | | | | | no behavioural accuracy or reaction time in the data available locally |
| H8 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | never_pooled |
| H8 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus; the delay-period optogenetic arm lies outside the stimulation definition |
| H8 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | macaque_pfc_microstimulation | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | never_pooled |
| H8 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | ram_ds005489_openloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | never_pooled |
| H8 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | never_pooled |
| H8 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H8 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 10: region or site to effect

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H4 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | H4|human|intracranial_field_potentials|sternberg_item_recognition|link10 |
| H4 | dandi_000004 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 59 | 0.358/0.030 | H4|human|units|new_old_recognition|link10 |
| H4 | dandi_000469 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 20 | 0.591/0.098 | H4|human|units|sternberg_item_recognition|link10 |
| H4 | dandi_000574 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis; depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.066 | H4|human|units|sternberg_item_recognition|link10 |
| H4 | dandi_000673 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 5 | 0.963/0.109 | H4|human|units|sternberg_item_recognition|link10 |
| H4 | dandi_001187 | frozen | P | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | patient | 39 | 0.436/0.036 | H4|human|units|sternberg_item_recognition|link10 |
| H4 | ds004752 | frozen | Y | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 15 | 0.669/0.048 | H4|human|depth_field_potentials|sternberg_item_recognition|link10 |
| H4 | ds005034 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | ds006848 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | H4|human|scalp_eeg|visual_retention_with_phase_locked_stimulation|link10 |
| H4 | inagaki_alm5 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | kai_miller_nback | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | macaque_pfc_microstimulation | not_askable | Y | | | | | | | one recording area (dorsolateral prefrontal cortex by design); site effects belong to the stimulation-site hypothesis |
| H4 | panichello_2024 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | pfc3 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | pfc4 | not_askable | P | | | | | | | the population state needs at least 8 units in the region-session and this corpus records at most 7 units per session |
| H4 | ram_ds005489_openloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | H4|human|depth_field_potentials|delayed_free_recall|link10 |
| H4 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | H4|human|depth_field_potentials|delayed_free_recall|link10 |
| H4 | watters_2026 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H4 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H7 | alagapan_phase_stimulation | frozen | P | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | intracranial_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics | patient | 3 | /0.159 | never_pooled |
| H7 | dandi_000004 | not_askable | P | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | ds004752 | not_askable | Y | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | ds005034 | not_askable | N | | | | | | | no region or site variation within the corpus |
| H7 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | haslacher_clam_tacs | frozen | P | accuracy (correct and error trials) | all_contacts_of_the_region_all_bands | scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 46 | 0.403/0.021 | never_pooled |
| H7 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | macaque_pfc_microstimulation | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis | session | 11 | 0.758/ | never_pooled |
| H7 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | pfc4 | not_askable | P | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | ram_ds005489_openloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | never_pooled |
| H7 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | never_pooled |
| H7 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |
| H7 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus; the stimulation-site effect needs stimulation |

## Link 11: dose to effect

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H7 | alagapan_phase_stimulation | not_askable | P | | | | | | | fewer than 6 patients (3 patients, one stimulation frequency each); the dose slope needs at least 6 subjects with 2 or more amplitude levels |
| H7 | dandi_000004 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | dandi_000469 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | dandi_000574 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | dandi_000673 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | dandi_001187 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | ds004752 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | ds005034 | not_askable | N | | | | | | | no dose variation: amplitude, frequency and duration are fixed |
| H7 | ds006848 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | haslacher_clam_tacs | not_askable | N | | | | | | | no dose variation: amplitude, frequency and duration are fixed |
| H7 | inagaki_alm5 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | kai_miller_nback | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | macaque_pfc_microstimulation | not_askable | N | | | | | | | no dose variation: amplitude, frequency and duration are fixed |
| H7 | panichello_2024 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | pfc3 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | pfc4 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | ram_ds005489_openloop | frozen | Y | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 37 | 0.447/ | never_pooled |
| H7 | ram_ds005557_closedloop | frozen | P | recall of the word (0/1) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, gaussian_process_factor_analysis | patient | 16 | 0.651/ | never_pooled |
| H7 | watters_2026 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | no stimulation in this corpus |
| H7 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | no stimulation in this corpus |

## Link 12: recording type to recording type

| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |
|---|---|---|---|---|---|---|---|---|---|---|
| H10 | alagapan_phase_stimulation | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | dandi_000004 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | dandi_000469 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | dandi_000574 | frozen | Y | log reaction time (correct trials only) | all_units | units: demixed_principal_components, gaussian_process_factor_analysis; depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.078 | never_pooled |
| H10 | dandi_000673 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | dandi_001187 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | ds004752 | frozen | Y | log reaction time (correct trials only) | all_contacts_of_the_region_all_bands | depth_field_potentials: demixed_principal_components, principal_components, factor_analysis, gaussian_process_factor_analysis, recurrent_switching_linear_dynamics; scalp_eeg: demixed_principal_components, gaussian_process_factor_analysis | patient | 9 | 0.816/0.078 | never_pooled |
| H10 | ds005034 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | ds006848 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | haslacher_clam_tacs | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | inagaki_alm5 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | kai_miller_nback | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | macaque_pfc_microstimulation | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | panichello_2024 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | pfc3 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | pfc4 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | ram_ds005489_openloop | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | ram_ds005557_closedloop | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | watters_2026 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | wolff_eeg_impulse experiment_1 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |
| H10 | wolff_eeg_impulse experiment_2 | not_askable | N | | | | | | | level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test |

