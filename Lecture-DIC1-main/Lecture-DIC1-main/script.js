const container = document.getElementById('space-container');
const btn = document.getElementById('gravity-btn');

// 預設是無重力狀態 (Antigravity is ON)
let isAntigravity = true;

btn.addEventListener('click', () => {
    isAntigravity = !isAntigravity;

    if (!isAntigravity) {
        // 關閉無重力 = 開啟重力 (東西往下掉)
        container.classList.add('gravity-on');
        btn.textContent = 'Gravity: ON (Click to restore Antigravity)';
        btn.style.borderColor = '#ef4444'; // 變成紅色警告
        btn.style.color = '#ef4444';
    } else {
        // 恢復無重力 (東西飄回來)
        container.classList.remove('gravity-on');
        btn.textContent = 'Gravity: OFF (Click to turn ON)';
        btn.style.borderColor = '#38bdf8'; // 恢復太空藍
        btn.style.color = '#38bdf8';
    }
});