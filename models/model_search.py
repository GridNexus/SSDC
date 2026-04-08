from torch.nn import init

from models.cmp import CMP
from models.pose import Block, ConvExpandReduce


class Search(CMP):
    def __init__(self, config):
        super().__init__(config,)

        self.be_hard = config.get('be_hard', False)
        self.be_pose_img = config.get('be_pose_img', False)
        self.be_pose_conv = config.get('pose_conv', False)
        if self.be_pose_img:
            self.pose_block = Block()
            self.init_params.extend(['pose_block.' + n for n, _ in self.pose_block.named_parameters()])
            if self.be_pose_conv:
                print('pose_conv')
                self.pose_conv = ConvExpandReduce()
                self.init_params.extend(['pose_conv.' + n for n, _ in self.pose_conv.named_parameters()])


    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, text_ids_eda=None, text_atts_eda=None,
                pose=None, hard_i=None, hard_i_pose=None, hard_text_ids=None, hard_text_atts=None,
                ):

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        if self.be_pose_img:
            if self.be_pose_conv:
                pose = self.pose_conv(pose)

            pose, _ = self.get_vision_embeds(pose)
            image_embeds = self.pose_block(image_embeds, pose)

        image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                          text_embeds, text_atts, text_feat, idx=idx)

        # eda
        text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
        text_feat_eda = self.get_text_feat(text_embeds_eda)
        loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
        loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                              text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx, )
        loss_itc = loss_itc + 0.8 * loss_itc_eda
        loss_itm = loss_itm + 0.8 * loss_itm_eda

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts,
                                     masked_pos, masked_ids, )

        if self.be_hard:
            image_embeds_hard, image_atts_hard = self.get_vision_embeds(hard_i)
            text_embeds_hard = self.get_text_embeds(hard_text_ids, hard_text_atts)

            if self.be_pose_img:
                if self.be_pose_conv:
                    hard_i_pose = self.pose_conv(hard_i_pose)

                hard_pose, _ = self.get_vision_embeds(hard_i_pose)
                image_embeds_hard = self.pose_block(image_embeds_hard, hard_pose)

            loss_itm_hard = self.get_matching_loss_hard(image_embeds, image_atts, image_embeds_hard, image_atts_hard,
                                                        text_embeds, text_atts, text_embeds_hard, hard_text_atts)
            loss_itm = loss_itm + loss_itm_hard

        return loss_itc, loss_itm, loss_mlm
