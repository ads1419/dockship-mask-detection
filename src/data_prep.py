from fastai.vision import *

path = Path("D:/Datasets/COVID-19-mask-detection")
import xml.etree.ElementTree as ET


def read_content(xml_file):

    """
        Reads the content of an XML file of the format:

            <annotation>
                <filename>a1.png</filename>
                <object>
                    <name>face</name>
                    <bndbox>
                        <xmin>168</xmin>
                        <ymin>221</ymin>
                        <xmax>441</xmax>
                        <ymax>618</ymax>
                    </bndbox>
                </object>
            </annotation>

        Returns: 
            filename: name of the image file
            anno: [  [ [bb1], [bb2] ], [l1, l2]  ] for all annotations present in the image.
            
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    bbs = []
    labels = []
    filename = root.find("filename").text
    for boxes in root.iter("object"):
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        for name in boxes.findall("name"):
            labels.append(name.text)
        #             if name.text == "face":
        #                 labels.append(1)
        #             else:
        #                 labels.append(2)

        bb = [xmin, ymin, xmax, ymax]  # standard bbox format: top left bottom right
        bbs.append(bb)
    assert len(bbs) == len(labels)

    if os.path.splitext(filename)[1] == ".xml":
        # to deal with a weird edge case where some test files had image filenames ending in .xml
        filename = os.path.splitext(filename)[0] + ".jpg"

    return filename, [bbs, labels]


def get_anno(pths):

    """
    
        Utility function that takes in a list of directories to look for annotations.
        Function crawls through the directory tree and picks out all the xml files and concatenates their annotations together.
        
        Returns:
            ims: list of image filenames
            lbl_bbox: list of items returned by read_content()
            
    """

    ims = []
    lbl_bbox = []

    for pth in anno_pths:
        for dirpath, dirnames, filenames in os.walk(pth):
            for file in filenames:
                file = Path(file)
                if file.suffix == ".xml":
                    try:
                        name, bbs = read_content(dirpath / file)
                        ims.append(name)
                        lbl_bbox.append(bbs)
                    except:
                        print(f"{file} does not have an annotation")
                        continue

    assert len(ims) == len(lbl_bbox)
    return ims, lbl_bbox


anno_pths = [path / "training/annotations", path / "validation/annotations"]
ims, lbl_bbox = get_anno(anno_pths)

img2bbox = dict(zip(ims, lbl_bbox))
get_y_func = lambda o: img2bbox[o.name]


def get_data(bs=16, imsize=224):
    data = (
        ObjectItemList.from_folder(path, exclude="example")
        .split_by_folder(train="training", valid="validation")
        .label_from_func(get_y_func)
        .transform(
            get_transforms(max_zoom=1.0),
            resize_method=ResizeMethod.SQUISH,
            tfm_y=True,
            size=224,
        )
        .databunch(bs=16, collate_fn=bb_pad_collate)
    )
    # NORMALIZE
    return data
