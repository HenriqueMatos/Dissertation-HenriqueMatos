import statistics
import deep_person_reid.torchreid as torchreid
# import torchreid
import time
import sys
import os
from os import listdir
from os.path import isfile, join
import os.path as osp
from deep_person_reid.torchreid.data import ImageDataset


def do_Re_Identification(gallery_directory, query_directory, use_gpu=False):

    class NewDataset(ImageDataset):
        dataset_dir = 'Actual_Tracking'

        def __init__(self, root='', **kwargs):
            self.root = osp.abspath(osp.expanduser(root))
            self.dataset_dir = osp.join(self.root, self.dataset_dir)

            # All you need to do here is to generate three lists,
            # which are train, query and gallery.
            # Each list contains tuples of (img_path, pid, camid),
            # where
            # - img_path (str): absolute path to an image.
            # - pid (int): person ID, e.g. 0, 1.
            # - camid (int): camera ID, e.g. 0, 1.
            # Note that
            # - pid and camid should be 0-based.
            # - query and gallery should share the same pid scope (e.g.
            #   pid=0 in query refers to the same person as pid=0 in gallery).
            # - train, query and gallery share the same camid scope (e.g.
            #   camid=0 in train refers to the same camera as camid=0
            #   in query/gallery).

            gallery_data = []
            query_data = []

            for path, subdirs, files in os.walk(gallery_directory):
                for name in files:
                    # print(os.path.join(path, name))
                    gallery_data.append((os.path.join(path, name), 1, 1))
            print(gallery_data)
            for path, subdirs, files in os.walk(query_directory):
                for name in files:
                    # print(os.path.join(path, name))
                    query_data.append((os.path.join(path, name), 1, 0))

            train = query_data
            query = query_data
            gallery = gallery_data
            # print(gallery_data)

            super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    torchreid.data.datasets.__image_datasets['new_dataset'] = NewDataset

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='new_dataset'
    )

    # torchreid.models.show_avai_models()

    model = torchreid.models.build_model(
        name="osnet_ain_x0_25",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
        use_gpu=use_gpu
    )

    if use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        # scheduler=scheduler,
        label_smooth=True,
        use_gpu=use_gpu
    )

    IDsObjectList = engine.run(
        dist_metric="cosine",
        save_dir="log/Actual_tracking",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        # rerank=True,
        ranks=[],
        visrank_topk=10,
        test_only=True,
        visrank=True
    )
    # print(IDsObjectList)
    FinalID = None
    meanAcceptValue = 70
    medianAcceptValue = 70
    modeAcceptValue = 75
    max_mean_value = 0
    for key, value in IDsObjectList.items():
        mean = statistics.fmean(value)
        median = statistics.median(value)
        mode = statistics.mode(value)
        print(mean)
        print(median)
        print(mode)
        if mean > meanAcceptValue and median > medianAcceptValue and mode > modeAcceptValue:
            if mean > max_mean_value:
                max_mean_value = mean
                FinalID=key
                # os._exit(1)
    # time.sleep(5000000)
    return FinalID
                

# do_Re_Identification('./reid-data/intersect_gallery_cam1', './reid-data/intersect_gallery_cam0/64dir1')
