"""
方向性C 総合バリデーション結果サマリー（スコアカード出力）
"""
import sys, os, types
m = types.ModuleType('multitasking')
m.set_max_threads = lambda x: None; m.set_engine = lambda x: None
m.task = lambda *a, **k: (lambda f: f); m.wait_for_tasks = lambda: None
sys.modules['multitasking'] = m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 各テスト結果（ハードコード済み実行結果）
RESULTS = [
    # T1
    {"test": "T1-1", "name": "ETFバスケット vs 合成モデル 日次相関 ≥ 0.90",
     "result": "FAIL", "detail": "相関=0.724（実際の 3x ETFバスケットとの乖離が大きい）"},
    {"test": "T1-2", "name": "累積NAV 乖離率 ≤ 30%",
     "result": "FAIL", "detail": "乖離47.2%、合成モデルの CAGR(22.0%) > ETFバスケット(17.4%)"},
    {"test": "T1-3", "name": "合成モデルが ETFバスケット比 2倍未満（上方バイアスなし）",
     "result": "PASS", "detail": "ratio=1.26（過度なバイアスなし）"},

    # T2a
    {"test": "T2a-1", "name": "ボンド期間差（30yr vs 10yr）の CAGR 差 ≤ 10pp",
     "result": "FAIL", "detail": "差=22.5pp（30yr:22.5% vs 10yr:~0%）→ ボンド期間仮定が致命的影響"},

    # T2b
    {"test": "T2b-1", "name": "ウェイト変動 9パターン中 80%+ がベースライン CAGR 超え",
     "result": "PASS", "detail": "9/9 = 100%（ウェイト方向性は安定）"},
    {"test": "T2b-2", "name": "ウェイト変動の最低 CAGR ≥ 30%",
     "result": "PASS", "detail": "min=31.01%（崩壊なし）"},

    # T3
    {"test": "T3-1", "name": "計画案(25%) が OOS Sharpe でベースラインを上回る",
     "result": "FAIL", "detail": "OOS: 0.8574 < 0.8854（逆転）"},
    {"test": "T3-2", "name": "計画案(25%) が OOS CAGR でベースラインを上回る",
     "result": "FAIL", "detail": "OOS: 20.91% < 21.75%（逆転）"},
    {"test": "T3-3", "name": "IS 最良(50%)の OOS Sharpe ランク ≥ 中位",
     "result": "PASS", "detail": "rank=3/6（辛うじて中位）"},
    {"test": "T3-4", "name": "OOS 最良 Sharpe が 0% 置換でない",
     "result": "FAIL", "detail": "OOS 最良 = 0%（Gold100%）→ IN-SAMPLE 過学習の典型的パターン"},

    # T4
    {"test": "T4-1", "name": "改善デケード ≥ 4/6",
     "result": "FAIL", "detail": "3/6: 1970s(✗) 1980s(✓) 1990s(✓) 2000s(✗) 2010s(✓) 2020s(✗)"},
    {"test": "T4-2", "name": "最悪デケードの Sharpe 低下 ≤ 0.15",
     "result": "PASS", "detail": "最大低下=-0.119（壊滅的失敗なし）"},
    {"test": "T4-3", "name": "1970年代 CAGR 差 ≥ -5pp（インフレ局面で壊滅なし）",
     "result": "PASS", "detail": "-3.26pp（許容範囲）"},

    # T5
    {"test": "T5-25", "name": "REIT プロキシ A/B/C 全バージョンで r=25% がベースライン超え",
     "result": "FAIL", "detail": "2004-2026 で VNQ実(B)版が同値止まり、SP500proxy(A)では下回る"},
    {"test": "T5-50", "name": "REIT プロキシ A/B/C 全バージョンで r=50% がベースライン超え",
     "result": "FAIL", "detail": "2004-2026 期間では全バージョンでベースラインを下回る"},

    # T6
    {"test": "T6-1", "name": "計画案(25%) 非補正 p < 0.10",
     "result": "FAIL", "detail": "p=0.613（完全に非有意）→ IS 改善は偶然範囲内"},
    {"test": "T6-2", "name": "95%CI 下限 ≥ -0.10（大幅悪化の証拠なし）",
     "result": "PASS", "detail": "CI_low=-0.0005（下方リスク小）"},
    {"test": "T6-3", "name": "最良比率(50%) の 95%CI 下限 ≥ -0.05",
     "result": "PASS", "detail": "CI_low=-0.0084（下方リスク小）"},
]

def main():
    p = sum(1 for r in RESULTS if r['result'] == 'PASS')
    f = sum(1 for r in RESULTS if r['result'] == 'FAIL')

    print("=" * 90)
    print("方向性C バリデーション 総合スコアカード")
    print("=" * 90)
    print(f"  総合: {p} PASSED / {f} FAILED / {p+f} 総テスト数")
    print()

    cats = [("T1: 合成モデルの実ETF整合性", "T1"),
            ("T2: パラメータ感度（ボンド期間・ウェイト）", "T2"),
            ("T3: OOS 検証（2021-2026）", "T3"),
            ("T4: 10年ごと一貫性", "T4"),
            ("T5: REITプロキシ汚染", "T5"),
            ("T6: 統計的有意性", "T6")]

    for cat_name, prefix in cats:
        cat_items = [r for r in RESULTS if r['test'].startswith(prefix)]
        cat_p = sum(1 for r in cat_items if r['result'] == 'PASS')
        cat_f = sum(1 for r in cat_items if r['result'] == 'FAIL')
        print(f"  ── {cat_name} ({cat_p}✅ / {cat_f}❌) ──")
        for r in cat_items:
            icon = "✅" if r['result'] == 'PASS' else "❌"
            print(f"    {icon} {r['test']}: {r['name']}")
            if r['result'] == 'FAIL':
                print(f"         → {r['detail']}")
        print()

    print("=" * 90)
    print("判定まとめ")
    print("=" * 90)
    print("""
  【結論】方向性C「まるごとレバレッジ → Goldスロット置換」は
          IN-SAMPLE では有望に見えたが、総合検証では **採用不可** と判定。

  ■ 重大な問題点（すべてが同一の根本原因を指す）:
    1. OOS 逆転（T3）: 2021-2026 でまるごとレバレッジ比率が高いほど性能低下
       → 全比率でベースライン（Gold100%）が OOS 最良。典型的な過学習パターン。
    2. 統計非有意（T6）: p=0.61（全く有意でない）
       → IS の CAGR+0.71%, Sharpe+0.023 は52年データでも偶然範囲内
    3. 合成モデル不正確（T1）: 実 3x ETFバスケットとの相関0.72, NAV乖離47%
       → 合成モデルが実際の商品の動作を正確に再現できていない
    4. ボンド期間リスク（T2a）: 30yr vs 10yr で CAGR が22.5pp 差
       → 実際のまるごとレバレッジが T-Note(10yr) を使うなら,
         本モデルは1974-1982 の金利上昇期に過度に楽観的な予測をしている
    5. 改善のデケード集中（T4）: 3/6デケードのみ改善
       → 1970s, 2000s, 2020s で悪化（現在の金利環境に近い期間での悪化）

  ■ 唯一残る証拠（採用理由にならない）:
    - ウェイト変動で方向性は一貫（T2b PASS）: ただし IS 内の話
    - 下方リスクは統計的に小さい（T6 CI PASS）: ただし点推定が非有意
    - 壊滅的な失敗なし（T4 PASS）: ただし改善も実証されていない

  ■ 真の原因:
    IS 改善の大部分は 1982-2020 の「38年間の債券強気相場」で債券3倍露出が
    大きく寄与したもの。この特定の歴史的レジームが IS に多く含まれているため
    バックテストでは改善に見えたが、2021-2026 の金利正常化局面でその恩恵が
    消え逆転した。

  ■ 推奨アクション:
    → 現状の Dyn 2x3x G0.5 を変更せず維持
    → まるごとレバレッジを実採用する場合は実ファンドデータ（2023年10月〜）で
      最低3年のOOS検証を実施してから判断
    → Gold スロットの改善を検討するなら「より流動性の高いGold1x ETF」など
      実績のある商品での代替を優先
""")

if __name__ == '__main__':
    main()
