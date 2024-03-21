# batch type
for modelpath in `cat $1`
do
    CUDA_VISIBLE_DEVICES=0 python test_cbn.py config/cbn/cbn_r18_ctw.py $modelpath
    #CUDA_VISIBLE_DEVICES=0 python test_cbn.py config/cbn/cbn_r18_msra.py $modelpath
    cd eval
    echo $modelpath
    sh eval_ctw.sh
    #sh eval_msra.sh
    cd ..
done
