import torch
from train_infogan_outcome import parse_args, get_tamoxifen_dataset
from train_infogan_outcome import DiscriminatorOutcome
from common import mutual_information, compute_mi_stats

def main():
    opt = parse_args()
    cuda = True
    train = False

    skip_study = opt.skip_study
    if skip_study == -1:
        skip_study = None
    additional_columns = ['study_ID', 'pam_coincide']
    train_set, test_set = get_tamoxifen_dataset(opt, additional_columns=additional_columns)
    if train:
        print("computing resulst on Train set")
        test_set, train_set = train_set, test_set
    else:
        print("computing resulst on Test set")

    pam50 = test_set.features.pam_coincide
    study = test_set.features.study_ID
    outcome = test_set.labels

    p_pam, p_outcome, p_pam_outcome = compute_mi_stats(pam50, outcome)
    mi = mutual_information(p_pam_outcome, p_pam, p_outcome)
    print("P(pam50): ", p_pam)
    print("P(outcome): ", p_outcome)
    print('MI(pam50, outcome): ', mi)

    p_study, p_outcome, p_study_outcome = compute_mi_stats(study, outcome)
    mi = mutual_information(p_study_outcome, p_study, p_outcome)
    print('MI(study, outcome): ', mi)
    if hasattr(opt, 'disc_path') and opt.disc_path:
        print('loading weights from ' + opt.disc_path)
        size = train_set.features.shape[1] - 2 # substract addional columns
        discriminator = DiscriminatorOutcome(opt, size, train_set.binary, train_set.continious)
        discriminator.load_state_dict(torch.load(opt.disc_path), strict=False)
        discriminator.eval()
        code = discriminator.compute_code(torch.as_tensor(test_set.features.drop(columns=['study_ID', 'pam_coincide']).to_numpy()).float()).argmax(dim=1)
        p_study, p_code, p_study_code = compute_mi_stats(study, code.cpu().numpy())
        mi = mutual_information(p_study_code, p_study, p_code)
        print("P(code): ", p_code)
        print('MI(study, code): ', mi)
        p_code, p_outcome, p_code_outcome = compute_mi_stats(code.cpu().numpy(), outcome)
        mi = mutual_information(p_code_outcome, p_code, p_outcome)
        print('MI(code, outcome): ', mi)
    else:
        print('\nWARNING  descriptor path not defined')



if __name__ == '__main__':
    main()
