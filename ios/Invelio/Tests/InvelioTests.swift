import XCTest
@testable import Invelio

final class InvelioTests: XCTestCase {
    func testStockDecoding() throws {
        let json = """
        {"ticker": "BBCA", "name": "Bank Central Asia", "price": 9875.0, "change": 1.25}
        """.data(using: .utf8)!

        let stock = try JSONDecoder().decode(Stock.self, from: json)
        XCTAssertEqual(stock.ticker, "BBCA")
        XCTAssertEqual(stock.price, 9875.0)
    }

    func testHoldingLotCalculations() throws {
        let lot = HoldingLot(
            ticker: "BBCA.JK",
            stockName: "Bank Central Asia",
            market: "IDX",
            currency: "IDR",
            pricePerShare: 10000,
            totalInvested: 5000000
        )

        XCTAssertEqual(lot.ticker, "BBCA.JK")
        XCTAssertEqual(lot.shares, 500.0)
        XCTAssertEqual(lot.lots, 5.0)
        XCTAssertEqual(lot.symbol, "BBCA")
    }

    func testLotBasedPortfolioCalculation() throws {
        // Example from user: price 1 share = 6000, 1 lot => 100 shares, portfolio = 600,000
        let lot = HoldingLot(
            ticker: "BBCA.JK",
            stockName: "Bank Central Asia",
            market: "IDX",
            currency: "IDR",
            pricePerShare: 6000,
            lots: 1.0
        )

        XCTAssertEqual(lot.lots, 1.0)
        XCTAssertEqual(lot.shares, 100.0)
        XCTAssertEqual(lot.totalInvested, 600000.0)
    }

    func testPurchaseFormEntryLotInput() throws {
        let entry = PurchaseFormEntry(
            priceInput: "6000",
            lotInput: "1"
        )

        XCTAssertEqual(entry.price, 6000.0)
        XCTAssertEqual(entry.lots, 1.0)
        XCTAssertEqual(entry.shares, 100.0)
        XCTAssertEqual(entry.total, 600000.0)
        XCTAssertEqual(entry.formattedShares, "100")
        XCTAssertEqual(entry.formattedLots, "1")
    }

    // MARK: - Markdown Table Parser Tests

    func testMarkdownTableParserPlainText() throws {
        let text = "Halo, ini adalah pesan biasa tanpa tabel sama sekali."
        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 1)
        if case .text(let content) = blocks.first?.content {
            XCTAssertEqual(content, text)
        } else {
            XCTFail("Expected text block")
        }
    }

    func testMarkdownTableParserBasicTable() throws {
        let text = """
        | Indikator | BBCA | BMRI |
        | :--- | :---: | ---: |
        | P/E Ratio | 20.5x | 11.2x |
        | PBV | 4.2x | 2.1x |
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 1)

        guard case .table(let table) = blocks.first?.content else {
            XCTFail("Expected table block")
            return
        }

        XCTAssertEqual(table.headers, ["Indikator", "BBCA", "BMRI"])
        XCTAssertEqual(table.alignments, [.leading, .center, .trailing])
        XCTAssertEqual(table.rows.count, 2)
        XCTAssertEqual(table.rows[0].cells, ["P/E Ratio", "20.5x", "11.2x"])
        XCTAssertEqual(table.rows[1].cells, ["PBV", "4.2x", "2.1x"])
    }

    func testMarkdownTableParserMixedContent() throws {
        let text = """
        Berikut perbandingan kedua saham:

        | Metrik | Nilai |
        | --- | --- |
        | ROE | 21% |
        | Deviden | 4% |

        BMRI memiliki yield deviden yang lebih tinggi.
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 3)

        if case .text(let intro) = blocks[0].content {
            XCTAssertTrue(intro.contains("Berikut perbandingan"))
        } else {
            XCTFail("Expected intro text")
        }

        if case .table(let table) = blocks[1].content {
            XCTAssertEqual(table.headers, ["Metrik", "Nilai"])
            XCTAssertEqual(table.rows.count, 2)
        } else {
            XCTFail("Expected table")
        }

        if case .text(let outro) = blocks[2].content {
            XCTAssertTrue(outro.contains("BMRI memiliki yield"))
        } else {
            XCTFail("Expected outro text")
        }
    }

    func testMarkdownTableParserNoOuterPipes() throws {
        let text = """
        Indikator | BBCA | BMRI
        --- | --- | ---
        P/E | 20x | 11x
        PBV | 4x | 2x
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 1)

        guard case .table(let table) = blocks.first?.content else {
            XCTFail("Expected table block")
            return
        }

        XCTAssertEqual(table.headers, ["Indikator", "BBCA", "BMRI"])
        XCTAssertEqual(table.rows.count, 2)
        XCTAssertEqual(table.rows[0].cells, ["P/E", "20x", "11x"])
    }

    func testMarkdownTableParserMultipleTables() throws {
        let text = """
        Tabel 1:
        | A | B |
        | --- | --- |
        | 1 | 2 |

        Tabel 2:
        | C | D |
        | --- | --- |
        | 3 | 4 |
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 4)

        if case .table(let table1) = blocks[1].content {
            XCTAssertEqual(table1.headers, ["A", "B"])
            XCTAssertEqual(table1.rows[0].cells, ["1", "2"])
        } else {
            XCTFail("Expected table 1")
        }

        if case .table(let table2) = blocks[3].content {
            XCTAssertEqual(table2.headers, ["C", "D"])
            XCTAssertEqual(table2.rows[0].cells, ["3", "4"])
        } else {
            XCTFail("Expected table 2")
        }
    }

    func testMarkdownTableParserPaddingRows() throws {
        let text = """
        | Kolom 1 | Kolom 2 | Kolom 3 |
        | --- | --- | --- |
        | Satu | Dua |
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 1)

        guard case .table(let table) = blocks.first?.content else {
            XCTFail("Expected table")
            return
        }

        XCTAssertEqual(table.rows[0].cells, ["Satu", "Dua", ""])
    }

    func testMarkdownTableParserNonTableWithPipes() throws {
        let text = """
        Pilihan opsi adalah A | B | C.
        Ini kalimat biasa tanpa baris delimiter.
        """

        let blocks = MarkdownTableParser.parse(text: text)
        XCTAssertEqual(blocks.count, 1)
        if case .text(let content) = blocks.first?.content {
            XCTAssertEqual(content, text)
        } else {
            XCTFail("Expected text block, not table")
        }
    }
}
