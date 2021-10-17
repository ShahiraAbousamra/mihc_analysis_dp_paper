cd /gpfs/projects/KurcGroup/sabousamra/multiplex1.0/NNFramework_Pytorch_external_call

CUDA_VISIBLE_DEVICES='7' nohup  python ./external_test.py ../NNFramework_PyTorch/config_multiplex_ae1.0_test/config_multiplex-ae_rgb2stains_simple-w-mp3_nocrop_inv_aux_task_input_stain_wsi2_test_96_patches.ini 0 &


CUDA_VISIBLE_DEVICES='6' nohup  python ./external_test.py ../NNFramework_PyTorch/config_multiplex_ae1.0_test/config_multiplex-ae_rgb2stains_simple-w-mp3_nocrop_inv_softmax_aux_task_input_stain_wsi2_test_96patches.ini 0 &


