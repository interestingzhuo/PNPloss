import datasampler.class_random_sampler


def select(sampler, opt, image_dict, image_list=None, **kwargs):
    if 'random' in sampler and 'class' in sampler:
            sampler_lib = class_random_sampler
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(sampler))

    sampler = sampler_lib.Sampler(opt,image_dict=image_dict,image_list=image_list)

    return sampler
