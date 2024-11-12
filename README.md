# FallLoader
This Project loads different datasets for Human Fall recognition in the same format for training tasks.

# Todo
Currently work in Progress, datasets are to be added and planned features: 
## Datasets
These datasets are considered to be added
- [x] Multiple Cameras Fall 
- [x] OOPS (Filtered)
- [x] EDF
- [x] OCCU
- [x] CAUCAFall
- [x] UR Fall
- [ ] Le2i
- [ ] UP Fall
- [ ] FPDS
- [ ] CMDFall (in progress)
- [ ] MUVIM
- [ ] IASLAB-RGBD

## Functionality
- [x] Load videos
    - [x] Load videos from frames images/binaries
        - [ ] test
    - [x] Load videos from rosbags
        - [ ] test
- [x] Load Labels: Time or Frame of Fall (if available)
- [x] Load Multiple falls in one video
- [x] Train-Test-Validate splits
    - [ ] Keep original split if available
- [x] Select subset of clips per dataset
- [ ] Download dataset from sources