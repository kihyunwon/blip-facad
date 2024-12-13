# Fashion Captioning Dataset
# https://arxiv.org/abs/2008.02693
# https://github.com/xuewyang/Fashion_Captioning

import gdown

captioning_dir = "dataset/fashion-captioning/"
# 아래 링크로 공유된 파일들만 다운로드 **약 198GB**
links = "https://drive.google.com/file/d/1rK6dZE7Cx4uB8Lju1uhqWWQL7DGf6r3-/view?usp=drive_link, https://drive.google.com/file/d/1lEQ8pV9tArY9hYAF5EyCDXcgkBHENIWw/view?usp=drive_link, https://drive.google.com/file/d/1qvWyJ8RL-nxMFPKav-ZUK3DsgRXeyhpQ/view?usp=drive_link, https://drive.google.com/file/d/14CUJ5k7LYtC7OU6QGB3gSjWirenCbMRW/view?usp=drive_link, https://drive.google.com/file/d/1gNn9x-jjVi9LD2RtKevz2-f8hBKg222r/view?usp=drive_link, https://drive.google.com/file/d/1zdyfclaN684Ah4aY5qKfaR4sxz1zcEB3/view?usp=drive_link, https://drive.google.com/file/d/1C8HYEgExCeR0co9h9mSpRro1D6IGMwd2/view?usp=drive_link, https://drive.google.com/file/d/1vl8VXVw6fpLB0SlwN4-bnYQbw1qp9D2E/view?usp=drive_link, https://drive.google.com/file/d/1IgmsUGgNr8zY8XOr-4FzbWjNs37GJ6GD/view?usp=drive_link, https://drive.google.com/file/d/1wqR5N65I0_eC9Z_0IFEAmZ-doyrs79zL/view?usp=drive_link, https://drive.google.com/file/d/1BMq4CHTtj24doEuecvIPR6sB2a86Ayq-/view?usp=drive_link, https://drive.google.com/file/d/1E0YZVFSSw_CdIX7tcEwKLn5mk1GCIHGn/view?usp=drive_link, https://drive.google.com/file/d/1D6KUavVWFtDRtGYczZPADJHD3MQkBr8b/view?usp=drive_link, https://drive.google.com/file/d/122NaXTWtIIn6_KeVhctqO01QQFpArWKN/view?usp=drive_link, https://drive.google.com/file/d/1Wkk7wv5gB7ZdpBMP-JKiEP9F0o0MHXvq/view?usp=drive_link, https://drive.google.com/file/d/1Sm0_DwoQWrDZTlKSqI1X2fP6ic0Apltd/view?usp=drive_link, https://drive.google.com/file/d/13DETLRw6ABqVh_Qay-4q9Rix8FWgy3rn/view?usp=drive_link, https://drive.google.com/file/d/1EFB_cCbTa-Dzxgsp9ZiK4nzQu4ZwrTUR/view?usp=drive_link, https://drive.google.com/file/d/1b6n8ijzpV6ylXrXZRx8em0NDzn9FdPC8/view?usp=drive_link, https://drive.google.com/file/d/14UEcwCgbTOYobPSJyoGwSRdAdjWQxieX/view?usp=drive_link"
link_list = links.split(", ")
f = lambda s: s.replace('https://drive.google.com/file/d/', '').replace('/view?usp=drive_link', '')
link_ids = list(map(f, link_list))

for l_id in link_ids:
    gdown.download(id=l_id, output=captioning_dir)
