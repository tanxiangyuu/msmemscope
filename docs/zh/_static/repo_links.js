// 在页面加载完成后执行
window.addEventListener('DOMContentLoaded', function() {
    // 创建链接容器
    var linksContainer = document.createElement('div');
    linksContainer.className = 'project-links';

    // 创建GitCode仓库链接
    var gitcodeLink = document.createElement('a');
    gitcodeLink.href = 'https://gitcode.com/Ascend/msmemscope'; // GitCode仓库地址
    gitcodeLink.target = '_blank';
    gitcodeLink.className = 'project-link';
    gitcodeLink.innerHTML = '<i class="fa fa-git"></i> GitCode';

    // 添加链接到容器
    linksContainer.appendChild(gitcodeLink);

    // 查找放置链接的位置 - 在project信息下方
    var projectNameElement = document.querySelector('.wy-side-nav-search > a');
    if (projectNameElement) {
        projectNameElement.parentNode.appendChild(linksContainer);
    } else {
        // 如果找不到特定位置，则尝试添加到页面顶部
        var headerElement = document.querySelector('.wy-side-nav-search');
        if (headerElement) {
            headerElement.appendChild(linksContainer);
        }
    }
});