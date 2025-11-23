"""
测试A股数据抓取
用于验证yfinance能否正确获取中国股票数据
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("="*80)
print("🧪 测试A股数据抓取")
print("="*80)

# 测试股票列表
test_stocks = {
    '600089': '特变电工',
    '600362': '江西铜业',
    '000878': '云南铜业',
    '000858': '五粮液',
    '600519': '贵州茅台'
}

# 测试日期（最近一周）
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"\n📅 测试日期: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
print(f"📊 测试股票: {len(test_stocks)}只\n")

results = {
    'success': [],
    'failed': []
}

# 方法1: 不加后缀（会失败）
print("\n" + "="*80)
print("方法1: 不加交易所后缀（预期会失败）")
print("="*80)

for symbol, name in test_stocks.items():
    try:
        print(f"\n尝试抓取: {symbol} ({name})")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(data) > 0:
            print(f"  ✅ 成功: 获取 {len(data)} 条数据")
            print(f"  📈 价格范围: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
            results['success'].append(symbol)
        else:
            print(f"  ⚠️  返回空数据")
            results['failed'].append(symbol)
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        results['failed'].append(symbol)

print(f"\n方法1结果: 成功 {len(results['success'])}/{len(test_stocks)}")

# 方法2: 加上交易所后缀（正确方式）
print("\n" + "="*80)
print("方法2: 添加交易所后缀 .SS/.SZ（正确方式）")
print("="*80)

results2 = {
    'success': [],
    'failed': []
}

for symbol, name in test_stocks.items():
    try:
        # 根据代码添加后缀
        if symbol.startswith('6'):
            yahoo_symbol = f"{symbol}.SS"  # 上海
        elif symbol.startswith('0') or symbol.startswith('3'):
            yahoo_symbol = f"{symbol}.SZ"  # 深圳
        else:
            yahoo_symbol = symbol

        print(f"\n尝试抓取: {yahoo_symbol} ({name})")
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(data) > 0:
            print(f"  ✅ 成功: 获取 {len(data)} 条数据")
            print(f"  📈 收盘价范围: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
            print(f"  📊 成交量均值: {data['Volume'].mean():.0f}")
            results2['success'].append(symbol)

            # 显示最近3天数据
            print(f"  最近数据预览:")
            for idx, row in data.tail(3).iterrows():
                print(f"    {idx.strftime('%Y-%m-%d')}: 开{row['Open']:.2f} 高{row['High']:.2f} 低{row['Low']:.2f} 收{row['Close']:.2f}")
        else:
            print(f"  ⚠️  返回空数据")
            results2['failed'].append(symbol)
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        results2['failed'].append(symbol)

print("\n" + "="*80)
print("📊 最终测试结果")
print("="*80)
print(f"方法1 (无后缀): 成功 {len(results['success'])}/{len(test_stocks)}")
print(f"方法2 (有后缀): 成功 {len(results2['success'])}/{len(test_stocks)}")

if len(results2['success']) > 0:
    print("\n✅ 结论: 需要添加交易所后缀!")
    print("   - 上海交易所 (6开头): 添加 .SS")
    print("   - 深圳交易所 (0/3开头): 添加 .SZ")
    print("\n修复建议: 修改 StockDataFetcher 类，在抓取前自动添加后缀")
else:
    print("\n❌ 两种方法都失败，可能是网络或yfinance问题")

print("="*80)

# 测试市场指数
print("\n🔍 测试市场指数抓取")
print("="*80)

indices = {
    '^SSEC': '上证指数',
    '000001.SS': '上证指数(备用)',
    '399001.SZ': '深证成指'
}

for symbol, name in indices.items():
    try:
        print(f"\n尝试抓取: {symbol} ({name})")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(data) > 0:
            print(f"  ✅ 成功: 获取 {len(data)} 条数据")
            print(f"  📈 指数范围: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
        else:
            print(f"  ⚠️  返回空数据")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")

print("\n" + "="*80)
print("✅ 测试完成!")
print("="*80)
