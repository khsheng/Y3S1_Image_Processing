from KHS.pidinet.custom_pidinet_main import pidinet_main

type_of_image = ["test", "train", "val"]
dataset_source="pcb-defect-median-opened-claheRGB"
new_dataset_name="pcb-defect-pidinet-median-opened-claheRGB(edge_id=3)"

for img_type in type_of_image:
    pidinet_main(type_of_image=img_type, dataset_source=dataset_source, new_dataset_name=new_dataset_name)
