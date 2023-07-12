# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [1.0.0-alpha.0](https://github.com/UynajGI/mcmc_statphys/compare/v0.4.3-20230517...v1.0.0-alpha.0) (2023-07-12)


### ⚠ BREAKING CHANGES

* 🧨 createModel!
* 🧨 Strauss模型！
* 🧨 Tempering类！
* 🧨 RFising类！
* 🧨 Kawasaki算法！
* 🧨 Demon方法！
* 🧨 新增数据保存msdt!
* 🧨 新method模组！
* 🧨 删除draw!
* 🧨 SKmodel!
* 🧨 WangLandau算法
* 🧨 删除draw模块！
* 🧨 新增 Parallel
* 🧨 删除了analysis模块！
* 🧨 iter_data 变为 data 属性

### Features

* 🎸 接受率增加heat-bath类别 ([f14047b](https://github.com/UynajGI/mcmc_statphys/commit/f14047b5e0da00a129ae9c3612d9e05d76c3ce35))
* 🎸 添加maxenergy属性 ([3564745](https://github.com/UynajGI/mcmc_statphys/commit/356474502b881e46611c48d489d4383fa0e21cc1))
* 🎸 添加u4计算，添加起始点分析数据功能 ([7baa8da](https://github.com/UynajGI/mcmc_statphys/commit/7baa8da4799f0916119d9fd36a7381aebdbe4592))
* 🎸 添加WangLandau算法 ([56c3c05](https://github.com/UynajGI/mcmc_statphys/commit/56c3c05e4723977890c67027b830c0fd19583792))
* 🎸 新增 Parallel 类 ([cb3ecf8](https://github.com/UynajGI/mcmc_statphys/commit/cb3ecf8d44151d9b659d76b53340a42fb1819717))
* 🎸 新增几个势能函数 ([d506d68](https://github.com/UynajGI/mcmc_statphys/commit/d506d6823192ba949a4a7df0b9ca819fcef23040))
* 🎸 新增自回归系数autocorrelation方法 ([a643edd](https://github.com/UynajGI/mcmc_statphys/commit/a643edd02c33fe5ce547f77ce1ad0850cab00e6b))
* 🎸 新增Demon方法 ([73cc393](https://github.com/UynajGI/mcmc_statphys/commit/73cc3937e24cdc6b46cdff2e0a313a834a2d3d42))
* 🎸 新增Ice和NVT模型（未完善） ([c4a85da](https://github.com/UynajGI/mcmc_statphys/commit/c4a85da02bda96085d84a09fc8ea8fa74a49d18b))
* 🎸 新增Kawasaki ([ab94ab1](https://github.com/UynajGI/mcmc_statphys/commit/ab94ab15edbe6c2bd8a82a0d3dfc4e44c5f4a5c6))
* 🎸 新增RFising类 ([8a34265](https://github.com/UynajGI/mcmc_statphys/commit/8a34265263eddecd9e79eace9f312274c7c3c2d4))
* 🎸 新增SKmodel类 ([bf713cc](https://github.com/UynajGI/mcmc_statphys/commit/bf713ccb31aee6dc731075173b204e73d6689b1c))
* 🎸 新增Staurss模型 ([20735b3](https://github.com/UynajGI/mcmc_statphys/commit/20735b3230d330594919c77b50187576e3de5e2b))
* 🎸 在method增加setup_uid方法 ([42566f9](https://github.com/UynajGI/mcmc_statphys/commit/42566f98efac2ec725d350b9089c317d5a297fa0))
* 🎸 在method增加setup_uid方法 ([bab1dd7](https://github.com/UynajGI/mcmc_statphys/commit/bab1dd7e3ec0d824a388864e39a65a465ae93a74))
* 🎸 增加数据保存类型msdt ([275ff64](https://github.com/UynajGI/mcmc_statphys/commit/275ff64078927e8ded057a66454934d6af614d2b))
* 🎸 增加model模板生成方法createModel ([897d1d3](https://github.com/UynajGI/mcmc_statphys/commit/897d1d37339d6cd0604bf0cf95d06a030e94e786))
* 🎸 autocorrelation方法中添加integrated correlation time输出 ([cb4a846](https://github.com/UynajGI/mcmc_statphys/commit/cb4a846c0dbbe29077734031bc8d7f8f60d2fb22))


### Bug Fixes

* 🐛 修复了进度条bug ([9e547b3](https://github.com/UynajGI/mcmc_statphys/commit/9e547b32f10b2f587d428f78379b39a265e7d8e3))
* 🐛 修改相关时间计算 ([23c8b68](https://github.com/UynajGI/mcmc_statphys/commit/23c8b680c8a1d0a4fa0458398d8aecdfea329ec7))
* 🐛 修改skmodel能量计算 ([d3007eb](https://github.com/UynajGI/mcmc_statphys/commit/d3007ebc972d6e125d2a1e2229f0c9e027a84d43))
* 🐛 修改u4计算错误 ([9a84c1d](https://github.com/UynajGI/mcmc_statphys/commit/9a84c1d9ae3c17d41e6bb537b6b3ad6e25773d57))


* 💡 分析绘图方法独立为新模组 ([fb03186](https://github.com/UynajGI/mcmc_statphys/commit/fb03186fde9e7f4bcf61bed53a82f4b00397b834))
* 💡 删除 draw模块，集成至algorithm模块 ([9021411](https://github.com/UynajGI/mcmc_statphys/commit/9021411fcd6e0c19d57eb31453c9d8fdf01b6756))
* 💡 删除了analysis模块，将它集成到了algorithm模块里 ([71dc925](https://github.com/UynajGI/mcmc_statphys/commit/71dc9253192d100e2a6dec7371da358d7a5387bf))
* 💡 删除draw模块，集成进algrithm模块 ([c16b116](https://github.com/UynajGI/mcmc_statphys/commit/c16b116d9ca92b5671b0c053f35dc9c211001b1b))
* 💡 iter_data 变为 data 属性 ([317b5f2](https://github.com/UynajGI/mcmc_statphys/commit/317b5f24709085564a1b90d1fa26de1fa775702f))
* 💡 Parallel 类改为Tempering类 ([01d8bce](https://github.com/UynajGI/mcmc_statphys/commit/01d8bceb3398b80649872bf2d65889aad12f3263))

## [2.0.0-20230517](https://github.com/UynajGI/mcmc_statphys/compare/v1.0.0-20230517...v2.0.0-20230517) (2023-05-21)


### ⚠ BREAKING CHANGES

* 🧨 animate imshow
* 🧨 新moudle
* 🧨 删除了 algorithm 的 Simulation 和 ParameterSample 类
* 🧨 algorithm 新增 Anneal 类
* 🧨 algorithm 新增 Wolff 类
* 🧨 新增了一个类

### Features

* 🎸 algorithm 新增 Anneal 类 ([278287a](https://github.com/UynajGI/mcmc_statphys/commit/278287a7e0fd132ada866205339ccadb240eb59f))
* 🎸 algorithm 新增 Wolff 类 ([be21137](https://github.com/UynajGI/mcmc_statphys/commit/be21137daf8ff951fb86f9689e894d9988cd4491))
* 🎸 algorithm 方法中新增了 Metropolis 类 ([753972b](https://github.com/UynajGI/mcmc_statphys/commit/753972b232509f452a25671ed4451b47e807b291))
* 🎸 analysis 里添加 cv 方法（涨落） ([9859cba](https://github.com/UynajGI/mcmc_statphys/commit/9859cba34897ca9a5d3676ab23e2f49ef62c0c2e))
* 🎸 moudle 增加外部方法get_energy get_magnetization ([7184d2a](https://github.com/UynajGI/mcmc_statphys/commit/7184d2a5db80d42e375bf3af22521b0585ea8bac))
* 🎸 删除了 algorithm 的 Simulation 和 ParameterSample 类 ([954dbc6](https://github.com/UynajGI/mcmc_statphys/commit/954dbc69783dc9356bf57f2bc0f7ba543e0106a1))
* 🎸 增加进度条功能 ([935f43c](https://github.com/UynajGI/mcmc_statphys/commit/935f43c2701f5a08945a61a14c7922dcbbca772b))
* 🎸 新增了setspin方法 ([257754b](https://github.com/UynajGI/mcmc_statphys/commit/257754bbc4df6db8931a83cf4b29da4f65e12333))
* 🎸 新的moudle analysis 用于分析algorithm产生的数据 ([3e6a72e](https://github.com/UynajGI/mcmc_statphys/commit/3e6a72e1a8f279e703b09b851cb51f7053f46c97))
* 🎸 添加draw模块 ([4cc0368](https://github.com/UynajGI/mcmc_statphys/commit/4cc0368e31ddfaba66a8a821f49bbc310b5b2622))
* 🎸 添加了animate 和 imshow功能 ([391e850](https://github.com/UynajGI/mcmc_statphys/commit/391e8509ca687b270c05060f3a4d8c552e0d258d))
* 🎸 添加本质值分析方法spin2svd uid2svd ([c7b35cb](https://github.com/UynajGI/mcmc_statphys/commit/c7b35cb877ef4e8e51491a151561740da4527400))
* test ([59319d4](https://github.com/UynajGI/mcmc_statphys/commit/59319d4301cf6e3b85f45615b8615434c488773e))


### Bug Fixes

* .gitignore ([456ea7e](https://github.com/UynajGI/mcmc_statphys/commit/456ea7e7ceb553698a49c2b164297682cba06288))
* .gitignore ([a472ed3](https://github.com/UynajGI/mcmc_statphys/commit/a472ed38fd24d53f74bb3870172ec3641ee798a1))
* .gitignore ([ca06ef2](https://github.com/UynajGI/mcmc_statphys/commit/ca06ef28e792ac71cdbb1db5694e3c8c7ac77a48))
* .gitignore ([0e7d707](https://github.com/UynajGI/mcmc_statphys/commit/0e7d707fa70e8b352d2397166f3d314ebe2a0c5c))
* 🐛 修复 wolff和退火中的uid不返回的问题 ([d656b07](https://github.com/UynajGI/mcmc_statphys/commit/d656b07b03e3e78b6251807a409688ec095ea53d))
* 🐛 修复了per到total ([0b22294](https://github.com/UynajGI/mcmc_statphys/commit/0b22294f68f3c5180b342ee4cdaa3fa776820295))
* 🐛 修正了param 方法进度条错误 ([9043e97](https://github.com/UynajGI/mcmc_statphys/commit/9043e976ee4ecc6fea28e4d085c46b8982a16f10))
* 🐛 磁矩储存per改为total/return uid/修改了一处注释 ([de13172](https://github.com/UynajGI/mcmc_statphys/commit/de131721d393a669030da64566e28b8ed5cc1908))
* 更改项目结构 ([76b043c](https://github.com/UynajGI/mcmc_statphys/commit/76b043c2953dd308a1ef44d1d32d6c68f8ce10dc))


### Styling

* 💄 修改TODO ([abd002f](https://github.com/UynajGI/mcmc_statphys/commit/abd002fa5fd7c990c05264532848d1aea3de4f70))
* 💄 注释美化 ([5d5949d](https://github.com/UynajGI/mcmc_statphys/commit/5d5949d6f6bba8a49a6e8b3d6a52b5728e99d2f8))


### Code Refactoring

* 💡 __init__.py 添加导入设置 ([f093572](https://github.com/UynajGI/mcmc_statphys/commit/f093572748e16bce9e5322fe93295ac9111ce38d))
* 💡 ref ([dfc3c3c](https://github.com/UynajGI/mcmc_statphys/commit/dfc3c3c9c2a5dbc4d811e4d81f945363b890a795))
* 💡 self.rowmodel to self._rowmodel ([b87eacf](https://github.com/UynajGI/mcmc_statphys/commit/b87eacfa102bb065361248cd7a8dc43cfeb1556b))
* 💡 tqdm 调整 ([13c1f50](https://github.com/UynajGI/mcmc_statphys/commit/13c1f506a23019b8e5323fd820826f10ad7400ff))
* 💡 优化代码结构 ([0d108d2](https://github.com/UynajGI/mcmc_statphys/commit/0d108d2c8a2c804c459a92b06e44b11187fcf882))
* 💡 优化运行速度，修正英文错误 ([30fa937](https://github.com/UynajGI/mcmc_statphys/commit/30fa937a916bb5624da7ed6a97404e4e9a9fee41))
* 💡 修正了格式标注 ([f96073f](https://github.com/UynajGI/mcmc_statphys/commit/f96073f032e7d85bc733d539109711d197da5945))
* 💡 将model的dimension改为dim ([b1cb000](https://github.com/UynajGI/mcmc_statphys/commit/b1cb00064cbf8ec3f83de589d0616ac0bfd10811))
* 💡 将属性 total_ 删去 ([d580690](https://github.com/UynajGI/mcmc_statphys/commit/d5806908494df1cfa2ce05039e8af336aa34b244))
* 💡 由data输入变成algorithm输入 ([23f62ec](https://github.com/UynajGI/mcmc_statphys/commit/23f62ec4d5034bbee0354b232b6d10c0f959b354))
* 发布0.1.1 ([da58e48](https://github.com/UynajGI/mcmc_statphys/commit/da58e48870ec8b823cf8a7a4de9a6168fc8e2c20))


### Tests

* 💍 2023-5-16 15:55:04 ([c3d926a](https://github.com/UynajGI/mcmc_statphys/commit/c3d926a5e7c2acdf7e753a04634b4ced8a904300))
* 💍 2023-5-16 20:15:46 ([4df1890](https://github.com/UynajGI/mcmc_statphys/commit/4df1890ff5598c43ca4b24e9b8d040b82a72a1fd))
* 💍 2023-5-16 22:03:03 ([4c884cb](https://github.com/UynajGI/mcmc_statphys/commit/4c884cb1520eb27badf8ffd2100daf0070976d81))
* 💍 2023-5-21 10:31:30 ([d144f0f](https://github.com/UynajGI/mcmc_statphys/commit/d144f0f7a8155bb8fc1ef35868585dfd193c89c8))
* 💍 2023-5-21 11:39:27 ([e448253](https://github.com/UynajGI/mcmc_statphys/commit/e448253a8ff24626cc30a2408aa55fed2ea75692))
* 测试action ([4d580f6](https://github.com/UynajGI/mcmc_statphys/commit/4d580f63567b7992bc6b74fb966947f03f229d28))


### Docs

* ✏️ change .gitignore ([0dfb97f](https://github.com/UynajGI/mcmc_statphys/commit/0dfb97fdeae2bf47361e1889eaa6f132eb6d78df))
* ✏️ v0.2.1 的文档更新 ([f8972db](https://github.com/UynajGI/mcmc_statphys/commit/f8972db1c2c400c512472fb1242db46ac394ef8c))
* ✏️ v0.3.0更新文档 ([f4bd408](https://github.com/UynajGI/mcmc_statphys/commit/f4bd4085ee0addc620f848f4901d961ac7670308))
* ✏️ 修改需求注释 ([8c03165](https://github.com/UynajGI/mcmc_statphys/commit/8c03165535ffcb4f5b29e901a14bcb1f4d3ba481))
* ✏️ 修正版本号 ([228a1ca](https://github.com/UynajGI/mcmc_statphys/commit/228a1ca7ba616b7d9d79f6f2026679668ece2ec9))
* ✏️ 文档修正单词错误 ([224e79c](https://github.com/UynajGI/mcmc_statphys/commit/224e79cdf83dce0b62d95edcd5100a9654933465))
* ✏️ 文档更新 ([11c9bc1](https://github.com/UynajGI/mcmc_statphys/commit/11c9bc1d2c0eb4c13f432db4a1f89927c956bb26))
* ✏️ 文档更新 ([77dcaf8](https://github.com/UynajGI/mcmc_statphys/commit/77dcaf80a6b725ab2ac3538127d4371b7817084c))
* ✏️ 文档更新 ([e739f25](https://github.com/UynajGI/mcmc_statphys/commit/e739f250688263a842f35049d6c84b6848a164b2))
* ✏️ 文档更新 ([f583885](https://github.com/UynajGI/mcmc_statphys/commit/f583885de61f8532d065eac099b2e065cd009cc2))
* ✏️ 更新History ([4e08c56](https://github.com/UynajGI/mcmc_statphys/commit/4e08c56a68b10e1f87b294b04d9064a844f8b54d))
* ✏️ 更新setup版本号 ([d853ec4](https://github.com/UynajGI/mcmc_statphys/commit/d853ec4138b2c03283307692867df9eaed520f80))
* ✏️ 更新文档 ([572f25e](https://github.com/UynajGI/mcmc_statphys/commit/572f25ea2173311f06aa15bedd143287bdb42a02))
* ✏️ 补充v0.2.1的日期 ([eb45367](https://github.com/UynajGI/mcmc_statphys/commit/eb453670e67e61402c59ce2ec47371a775cca8ad))


### CI

* 🎡 修改自动流相关 ([0fd3cd7](https://github.com/UynajGI/mcmc_statphys/commit/0fd3cd73eb33cbc00eb2db3d8175986a973976c9))
* 🎡 增加自动流 ([9f05973](https://github.com/UynajGI/mcmc_statphys/commit/9f05973e2dc3c303d22fc0164090528bbf8f1512))
* 🎡 增加自动迭代工具 ([e6037d6](https://github.com/UynajGI/mcmc_statphys/commit/e6037d694c16e9882c36ba843ab9331d2c5f58bc))

## [1.0.0-20230517](https://github.com/UynajGI/mcmc_statphys/compare/v0.2.1-20230517...v1.0.0-20230517) (2023-05-21)


### ⚠ BREAKING CHANGES

* 🧨 animate imshow

### Features

* 🎸 analysis 里添加 cv 方法（涨落） ([9859cba](https://github.com/UynajGI/mcmc_statphys/commit/9859cba34897ca9a5d3676ab23e2f49ef62c0c2e))
* 🎸 moudle 增加外部方法get_energy get_magnetization ([7184d2a](https://github.com/UynajGI/mcmc_statphys/commit/7184d2a5db80d42e375bf3af22521b0585ea8bac))
* 🎸 添加了animate 和 imshow功能 ([391e850](https://github.com/UynajGI/mcmc_statphys/commit/391e8509ca687b270c05060f3a4d8c552e0d258d))
* 🎸 添加本质值分析方法spin2svd uid2svd ([c7b35cb](https://github.com/UynajGI/mcmc_statphys/commit/c7b35cb877ef4e8e51491a151561740da4527400))


### Bug Fixes

* 🐛 修复 wolff和退火中的uid不返回的问题 ([d656b07](https://github.com/UynajGI/mcmc_statphys/commit/d656b07b03e3e78b6251807a409688ec095ea53d))


### Code Refactoring

* 💡 tqdm 调整 ([13c1f50](https://github.com/UynajGI/mcmc_statphys/commit/13c1f506a23019b8e5323fd820826f10ad7400ff))
* 💡 优化代码结构 ([0d108d2](https://github.com/UynajGI/mcmc_statphys/commit/0d108d2c8a2c804c459a92b06e44b11187fcf882))
* 💡 优化运行速度，修正英文错误 ([30fa937](https://github.com/UynajGI/mcmc_statphys/commit/30fa937a916bb5624da7ed6a97404e4e9a9fee41))
* 💡 修正了格式标注 ([f96073f](https://github.com/UynajGI/mcmc_statphys/commit/f96073f032e7d85bc733d539109711d197da5945))


### CI

* 🎡 增加自动流 ([9f05973](https://github.com/UynajGI/mcmc_statphys/commit/9f05973e2dc3c303d22fc0164090528bbf8f1512))
* 🎡 增加自动迭代工具 ([e6037d6](https://github.com/UynajGI/mcmc_statphys/commit/e6037d694c16e9882c36ba843ab9331d2c5f58bc))


### Tests

* 💍 2023-5-21 10:31:30 ([d144f0f](https://github.com/UynajGI/mcmc_statphys/commit/d144f0f7a8155bb8fc1ef35868585dfd193c89c8))
* 💍 2023-5-21 11:39:27 ([e448253](https://github.com/UynajGI/mcmc_statphys/commit/e448253a8ff24626cc30a2408aa55fed2ea75692))


### Docs

* ✏️ v0.3.0更新文档 ([f4bd408](https://github.com/UynajGI/mcmc_statphys/commit/f4bd4085ee0addc620f848f4901d961ac7670308))
* ✏️ 修改需求注释 ([8c03165](https://github.com/UynajGI/mcmc_statphys/commit/8c03165535ffcb4f5b29e901a14bcb1f4d3ba481))
* ✏️ 修正版本号 ([228a1ca](https://github.com/UynajGI/mcmc_statphys/commit/228a1ca7ba616b7d9d79f6f2026679668ece2ec9))
* ✏️ 文档修正单词错误 ([224e79c](https://github.com/UynajGI/mcmc_statphys/commit/224e79cdf83dce0b62d95edcd5100a9654933465))
* ✏️ 文档更新 ([11c9bc1](https://github.com/UynajGI/mcmc_statphys/commit/11c9bc1d2c0eb4c13f432db4a1f89927c956bb26))
* ✏️ 文档更新 ([77dcaf8](https://github.com/UynajGI/mcmc_statphys/commit/77dcaf80a6b725ab2ac3538127d4371b7817084c))
* ✏️ 文档更新 ([e739f25](https://github.com/UynajGI/mcmc_statphys/commit/e739f250688263a842f35049d6c84b6848a164b2))
* ✏️ 文档更新 ([f583885](https://github.com/UynajGI/mcmc_statphys/commit/f583885de61f8532d065eac099b2e065cd009cc2))
* ✏️ 更新setup版本号 ([d853ec4](https://github.com/UynajGI/mcmc_statphys/commit/d853ec4138b2c03283307692867df9eaed520f80))
* ✏️ 更新文档 ([572f25e](https://github.com/UynajGI/mcmc_statphys/commit/572f25ea2173311f06aa15bedd143287bdb42a02))
