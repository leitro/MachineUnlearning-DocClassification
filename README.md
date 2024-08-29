# MachineUnlearning-DocClassification
Official Implementation for ICDAR2024 paper ["Machine Unlearning for Document Classification"](https://arxiv.org/pdf/2404.19031)


## Dataset

Please find the RVL-CDIP dataset [HERE](https://adamharley.com/rvl-cdip/).

Once you've acquired the dataset and placed it in your folder, be sure to update lines 11-12 roughly in the `dataset*.py` files accordingly.

* `dataset_all.py`: original data to train the document classifier (baseline).
* `dataset_unlearn_aug.py`: real retain and real forget sets for full/subset machine unlearning
* `dataset_unlearn_aug_randGen_forget_solo.py`: real retain and randomly generated forget sets for full/subset machine unlearning
* `dataset_unlearn_aug_mohaGen_forget_solo.py`: real retain and label-guided generated forget sets for full/subset machine unlearning

In the restricted unlearning scenario, we proposed to store only a subset of 10% of training set for unlearning experiments and introduced four selection strategies: random, top, bottom, and mix. Please find all the data in the folder `LABEL_SC`. In addition to the 10% subset, we also provided 5% and 1% subsets, which you can try as well. 


## Train the model

* Baseline document classifier: `python train_base.py`. We released our weights `rvl-41.model.87.93.scratch`.
* Machine unlearning using real forget set: `python train_unlearn_all_iter.py`
* Machine unlearning using randomly generated forget set: `python train_unlearn_all_iter_randGen_forget_solo.py`
* Machine unlearning using label-guided generated forget set: `python train_unlearn_all_iter_mohaGen_forget_solo.py`


Please note that we used the real retain set (either in full or subset mode) for all unlearning experiments. The machine unlearning experiments mentioned above were conducted with different types of forget sets: real, randomly generated, and label-guided generated.


## Citation

If you find our work helpful for your research or use it as a baseline model, please cite our paper as follows:

```bibtex
@inproceedings{kang2024machine,
  title={Machine Unlearning for Document Classification},
  author={Kang, Lei and Souibgui, Mohamed Ali and Yang, Fei and Gomez, Lluis and Valveny, Ernest and Karatzas, Dimosthenis},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2024},
  organization={Springer}
}
```
