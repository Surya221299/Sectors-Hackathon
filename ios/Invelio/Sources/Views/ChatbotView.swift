import SwiftUI
import SwiftData

// MARK: - Message Model
struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    let text: String
    let isUser: Bool
    let isFinished: Bool
    let timestamp: Date

    init(
        id: UUID = UUID(),
        text: String,
        isUser: Bool,
        isFinished: Bool = true,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.isUser = isUser
        self.isFinished = isFinished
        self.timestamp = timestamp
    }

    var displayText: String {
        guard !isUser else { return text }
        return text
            .replacingOccurrences(
                of: #"(?m)^#{1,6}\s*(.+)$"#,
                with: "**$1**",
                options: .regularExpression
            )
            .replacingOccurrences(of: "###", with: "")
    }
}

// MARK: - View Model
@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isProcessing: Bool = false

    private var sessionId = UUID()
    private var streamTask: Task<Void, Never>?

    func send(_ query: String, promptOverride: String? = nil) {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isProcessing else { return }

        let userMessage = ChatMessage(text: trimmed, isUser: true, isFinished: true)
        messages.append(userMessage)
        inputText = ""
        isProcessing = true

        let botMessageId = UUID()
        messages.append(ChatMessage(id: botMessageId, text: "", isUser: false, isFinished: false))

        let payload = promptOverride ?? trimmed
        streamTask = Task {
            await streamFromAgent(query: payload, botMessageId: botMessageId)
        }
    }

    private func streamFromAgent(query: String, botMessageId: UUID) async {
        var accumulated = ""
        let stream = APIClient.shared.chatStream(
            message: query,
            sessionId: sessionId
        )

        do {
            for try await chunk in stream {
                accumulated += chunk
                updateBotMessage(id: botMessageId, text: accumulated, isFinished: false)
            }
        } catch {
            print("[ChatBot] Stream error: \(error)")
            if accumulated.isEmpty {
                accumulated = "Error: \(error.localizedDescription)"
            }
        }
        withAnimation(.easeOut(duration: 0.3)) {
            updateBotMessage(id: botMessageId, text: accumulated, isFinished: true)
        }
        isProcessing = false
    }

    private func updateBotMessage(id: UUID, text: String, isFinished: Bool) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index] = ChatMessage(id: id, text: text, isUser: false, isFinished: isFinished)
    }

    func resetSession() {
        streamTask?.cancel()
        messages.removeAll()
        inputText = ""
        isProcessing = false
        sessionId = UUID()
    }
}

// MARK: - Markdown Table Models & Content
public struct MarkdownTableRow: Identifiable, Equatable {
    public let id: Int
    public let cells: [String]

    public init(id: Int, cells: [String]) {
        self.id = id
        self.cells = cells
    }
}

public struct MarkdownTable: Identifiable, Equatable {
    public let id: String
    public let headers: [String]
    public let alignments: [TextAlignment]
    public let rows: [MarkdownTableRow]

    public var rawRows: [[String]] {
        rows.map { $0.cells }
    }

    public init(
        id: String = UUID().uuidString,
        headers: [String],
        alignments: [TextAlignment],
        rows: [MarkdownTableRow]
    ) {
        self.id = id
        self.headers = headers
        self.alignments = alignments
        self.rows = rows
    }

    public init(
        id: String = UUID().uuidString,
        headers: [String],
        alignments: [TextAlignment],
        rawRows: [[String]]
    ) {
        self.id = id
        self.headers = headers
        self.alignments = alignments
        self.rows = rawRows.enumerated().map { MarkdownTableRow(id: $0.offset, cells: $0.element) }
    }
}

public struct ChatContentBlock: Identifiable, Equatable {
    public let id: String
    public let content: BlockContent

    public enum BlockContent: Equatable {
        case text(String)
        case table(MarkdownTable)
    }

    public init(id: String, content: BlockContent) {
        self.id = id
        self.content = content
    }
}

// MARK: - Markdown Table Parser
public enum MarkdownTableParser {
    public static func parse(text: String) -> [ChatContentBlock] {
        guard text.contains("|") else {
            return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? []
                : [ChatContentBlock(id: "block_0", content: .text(text))]
        }

        let normalizedText = text.replacingOccurrences(of: "\r\n", with: "\n")
        let rawLines = normalizedText.components(separatedBy: "\n")
        var blocks: [ChatContentBlock] = []
        var currentTextLines: [String] = []

        func flushTextIfNeeded() {
            guard !currentTextLines.isEmpty else { return }
            let textBlock = currentTextLines.joined(separator: "\n")
            if !textBlock.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                blocks.append(ChatContentBlock(id: "block_\(blocks.count)", content: .text(textBlock)))
            }
            currentTextLines.removeAll()
        }

        var i = 0
        while i < rawLines.count {
            let line = rawLines[i]

            // A table requires a header line with '|' followed by a delimiter line
            if i + 1 < rawLines.count,
               isPotentialTableRow(line),
               parseDelimiterRow(line) == nil,
               let alignments = parseDelimiterRow(rawLines[i + 1]) {

                let headerCells = splitTableRow(line)
                if !headerCells.isEmpty {
                    flushTextIfNeeded()

                    var normalizedAlignments = alignments
                    if normalizedAlignments.count < headerCells.count {
                        normalizedAlignments += Array(repeating: .leading, count: headerCells.count - normalizedAlignments.count)
                    } else if normalizedAlignments.count > headerCells.count {
                        normalizedAlignments = Array(normalizedAlignments.prefix(headerCells.count))
                    }

                    // Advance past header and delimiter line
                    i += 2

                    // Parse data rows
                    var dataRows: [MarkdownTableRow] = []
                    var rowIdx = 0
                    while i < rawLines.count {
                        let rowLine = rawLines[i]
                        let trimmed = rowLine.trimmingCharacters(in: .whitespaces)

                        // Empty line terminates table
                        if trimmed.isEmpty {
                            break
                        }

                        // Non-table line terminates table
                        if !isPotentialTableRow(rowLine) {
                            break
                        }

                        // Another delimiter row terminates table
                        if parseDelimiterRow(rowLine) != nil {
                            break
                        }

                        var rowCells = splitTableRow(rowLine)
                        if rowCells.count < headerCells.count {
                            rowCells += Array(repeating: "", count: headerCells.count - rowCells.count)
                        } else if rowCells.count > headerCells.count {
                            rowCells = Array(rowCells.prefix(headerCells.count))
                        }

                        dataRows.append(MarkdownTableRow(id: rowIdx, cells: rowCells))
                        rowIdx += 1
                        i += 1
                    }

                    let table = MarkdownTable(
                        id: "table_\(blocks.count)",
                        headers: headerCells,
                        alignments: normalizedAlignments,
                        rows: dataRows
                    )
                    blocks.append(ChatContentBlock(id: "block_\(blocks.count)", content: .table(table)))
                    continue
                }
            }

            currentTextLines.append(line)
            i += 1
        }

        flushTextIfNeeded()

        return blocks
    }

    private static func isPotentialTableRow(_ line: String) -> Bool {
        line.contains("|")
    }

    public static func splitTableRow(_ line: String) -> [String] {
        let escapedPlaceholder = "\u{FFF0}"
        let safeLine = line.replacingOccurrences(of: "\\|", with: escapedPlaceholder)
        var rawCells = safeLine.components(separatedBy: "|")

        // Drop empty leading cell if line started with '|'
        if let first = rawCells.first, first.trimmingCharacters(in: .whitespaces).isEmpty, rawCells.count > 1 {
            rawCells.removeFirst()
        }

        // Drop empty trailing cell if line ended with '|'
        if let last = rawCells.last, last.trimmingCharacters(in: .whitespaces).isEmpty, rawCells.count > 1 {
            rawCells.removeLast()
        }

        return rawCells.map {
            $0.replacingOccurrences(of: escapedPlaceholder, with: "|")
              .trimmingCharacters(in: .whitespaces)
        }
    }

    public static func parseDelimiterRow(_ line: String) -> [TextAlignment]? {
        guard line.contains("-") else { return nil }

        let cells = splitTableRow(line)
        guard !cells.isEmpty else { return nil }

        var alignments: [TextAlignment] = []

        for cell in cells {
            let trimmed = cell.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { return nil }

            let allowedCharacters = CharacterSet(charactersIn: "-:")
            guard CharacterSet(charactersIn: trimmed).isSubset(of: allowedCharacters) else {
                return nil
            }

            guard trimmed.contains("-") else { return nil }

            let startsWithColon = trimmed.hasPrefix(":")
            let endsWithColon = trimmed.hasSuffix(":")

            if startsWithColon && endsWithColon {
                alignments.append(.center)
            } else if endsWithColon {
                alignments.append(.trailing)
            } else {
                alignments.append(.leading)
            }
        }

        return alignments
    }
}

// MARK: - Markdown Content View
struct MarkdownContentView: View {
    let text: String

    private var blocks: [ChatContentBlock] {
        MarkdownTableParser.parse(text: text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(blocks) { block in
                switch block.content {
                case .text(let content):
                    Text(LocalizedStringKey(content))
                        .font(.system(size: 14.5, weight: .regular))
                        .foregroundStyle(Color.white.opacity(0.92))
                        .lineSpacing(4)
                case .table(let table):
                    MarkdownTableView(table: table)
                }
            }
        }
    }
}

// MARK: - Markdown Table View
struct MarkdownTableView: View {
    let table: MarkdownTable

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    // Header Row
                    headerRow

                    // Header Bottom Divider
                    Rectangle()
                        .fill(Color.white.opacity(0.14))
                        .frame(height: 1)

                    // Data Rows
                    ForEach(table.rows) { row in
                        if row.id > 0 {
                            Rectangle()
                                .fill(Color.white.opacity(0.06))
                                .frame(height: 0.5)
                        }
                        dataRow(row)
                    }
                }
                .background(Color(red: 21/255, green: 17/255, blue: 48/255))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Color.white.opacity(0.14), lineWidth: 1)
                )
            }

            if table.headers.count >= 3 {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.left.and.right")
                        .font(.system(size: 9))
                    Text("Geser untuk melihat semua kolom")
                        .font(.system(size: 10, weight: .regular))
                }
                .foregroundStyle(Color.white.opacity(0.42))
                .padding(.leading, 2)
                .padding(.top, 1)
            }
        }
        .padding(.vertical, 3)
    }

    private var headerRow: some View {
        HStack(spacing: 0) {
            ForEach(Array(table.headers.enumerated()), id: \.offset) { colIndex, header in
                headerCell(
                    text: header,
                    colIndex: colIndex,
                    width: columnWidth(for: colIndex)
                )
            }
        }
        .background(Color.white.opacity(0.08))
    }

    private func dataRow(_ row: MarkdownTableRow) -> some View {
        HStack(spacing: 0) {
            ForEach(table.headers.indices, id: \.self) { colIndex in
                let cellText = colIndex < row.cells.count ? row.cells[colIndex] : ""
                dataCell(
                    text: cellText,
                    rowIndex: row.id,
                    colIndex: colIndex,
                    width: columnWidth(for: colIndex)
                )
            }
        }
        .background(row.id.isMultiple(of: 2) ? Color.white.opacity(0.025) : Color.clear)
    }

    private func columnWidth(for colIndex: Int) -> CGFloat {
        let total = table.headers.count
        let headerText = colIndex < table.headers.count ? table.headers[colIndex] : ""

        let baseMin: CGFloat
        if total <= 2 {
            baseMin = colIndex == 0 ? 140 : 125
        } else if total == 3 {
            baseMin = colIndex == 0 ? 120 : 95
        } else {
            baseMin = colIndex == 0 ? 115 : 85
        }

        let estimatedWidth = CGFloat(headerText.count) * 7.5 + 24
        return max(baseMin, min(220, estimatedWidth))
    }

    private func frameAlignment(for colIndex: Int) -> Alignment {
        guard colIndex < table.alignments.count else { return .leading }
        switch table.alignments[colIndex] {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    private func textAlignment(for colIndex: Int) -> TextAlignment {
        guard colIndex < table.alignments.count else { return .leading }
        return table.alignments[colIndex]
    }

    @ViewBuilder
    private func headerCell(text: String, colIndex: Int, width: CGFloat) -> some View {
        HStack(spacing: 0) {
            Text(LocalizedStringKey(text))
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(Color.white)
                .multilineTextAlignment(textAlignment(for: colIndex))
                .lineLimit(2)
        }
        .frame(width: width, alignment: frameAlignment(for: colIndex))
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .overlay(alignment: .trailing) {
            if colIndex < table.headers.count - 1 {
                Rectangle()
                    .fill(Color.white.opacity(0.12))
                    .frame(width: 1)
            }
        }
    }

    @ViewBuilder
    private func dataCell(text: String, rowIndex: Int, colIndex: Int, width: CGFloat) -> some View {
        let isPlaceholder = text.trimmingCharacters(in: .whitespaces).isEmpty
        let displayContent = isPlaceholder ? "-" : text

        HStack(spacing: 0) {
            Text(LocalizedStringKey(displayContent))
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(isPlaceholder ? Color.white.opacity(0.35) : cellTextColor(for: text))
                .monospacedDigit()
                .multilineTextAlignment(textAlignment(for: colIndex))
                .lineLimit(2)
        }
        .frame(width: width, alignment: frameAlignment(for: colIndex))
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .overlay(alignment: .trailing) {
            if colIndex < table.headers.count - 1 {
                Rectangle()
                    .fill(Color.white.opacity(0.06))
                    .frame(width: 1)
            }
        }
    }

    private func cellTextColor(for rawText: String) -> Color {
        let clean = rawText
            .replacingOccurrences(of: "*", with: "")
            .trimmingCharacters(in: .whitespaces)
            .uppercased()

        if clean == "BUY" || clean == "BELI" || clean == "RECOMMENDED" {
            return .ProfitGreen
        }
        if clean == "SELL" || clean == "JUAL" || clean == "CAUTION" {
            return .LossRed
        }
        if clean.hasPrefix("+") && clean.hasSuffix("%") {
            return .ProfitGreen
        }
        if clean.hasPrefix("-") && clean.hasSuffix("%") {
            return .LossRed
        }
        return Color.white.opacity(0.90)
    }
}

// MARK: - Chat Bubble Row
struct ChatBubbleRow: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.isUser {
                Spacer(minLength: 44)
                userBubble
            } else {
                botBubble
                Spacer(minLength: 44)
            }
        }
    }

    private var userBubble: some View {
        Text(message.text)
            .font(.system(size: 15, weight: .regular))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 65/255, green: 55/255, blue: 135/255),
                                Color(red: 45/255, green: 38/255, blue: 95/255)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.white.opacity(0.15), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.2), radius: 6, x: 0, y: 3)
    }

    private var botBubble: some View {
        Group {
            if message.text.isEmpty {
                HStack(spacing: 8) {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.8)
                    Text("Analyzing...")
                        .font(.system(size: 14))
                        .foregroundStyle(Color.white.opacity(0.7))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.AICardBg)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.2), radius: 6, x: 0, y: 2)
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    // Message Content
                    MarkdownContentView(text: message.displayText)

                    if message.isFinished {
                        // Source Indicator
                        HStack(spacing: 5) {
                            Text("Source:")
                                .font(.system(size: 11.5, weight: .regular))
                                .foregroundStyle(Color.white.opacity(0.55))

                            HStack(spacing: 4.5) {
                                Image("sectors_logo")
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 12, height: 12)
                                Text("Sectors")
                                    .font(.system(size: 11.5, weight: .semibold))
                                    .foregroundStyle(Color.white.opacity(0.9))
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                Capsule()
                                    .fill(Color.white.opacity(0.08))
                            )
                            .overlay(
                                Capsule()
                                    .stroke(Color.white.opacity(0.12), lineWidth: 0.5)
                            )
                        }
                        .padding(.top, 2)
                        .transition(.opacity.combined(with: .move(edge: .top)))

                        // AI Disclaimer Banner (Under Source, Yellow with opacity, icon aligned with text start)
                        HStack(alignment: .top, spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 10.5))
                                .foregroundStyle(Color.PrimaryYellow.opacity(0.9))
                                .padding(.top, 1.5)
                            Text("Disclaimer: AI-generated content. For informational purposes only. Not financial or investment advice.")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(Color.PrimaryYellow.opacity(0.85))
                                .lineSpacing(2)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .fill(Color.PrimaryYellow.opacity(0.10))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .stroke(Color.PrimaryYellow.opacity(0.25), lineWidth: 0.8)
                        )
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.AICardBg)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.2), radius: 8, x: 0, y: 2)
            }
        }
    }
}

// MARK: - Main Chatbot View
struct ChatbotView: View {
    @Query(sort: \HoldingLot.buyDate, order: .reverse) private var holdingLots: [HoldingLot]
    @StateObject private var viewModel = ChatViewModel()
    @FocusState private var isInputFocused: Bool

    private let sampleSuggestions = [
        "My Stock Outlook",
        "My portfolio Risks",
        "Recommended Stocks",
        "Why Did My Stock Move?"
    ]

    private func enrichPromptIfNeeded(_ query: String) -> String {
        let lower = query.lowercased()
        let portfolioKeywords = [
            "portfolio", "portofolio", "holding", "my stock", "saham saya",
            "outlook", "risk", "risks", "why did my stock move"
        ]
        let isPortfolioQuery = portfolioKeywords.contains { lower.contains($0) }
        guard isPortfolioQuery, !holdingLots.isEmpty else {
            return query
        }

        var summary: [String: (name: String, shares: Double, invested: Double)] = [:]
        for lot in holdingLots {
            let t = lot.ticker.components(separatedBy: ".").first?.uppercased() ?? lot.ticker.uppercased()
            if summary[t] == nil {
                summary[t] = (name: lot.stockName, shares: 0, invested: 0)
            }
            summary[t]?.shares += lot.shares
            summary[t]?.invested += lot.totalInvested
        }

        let holdingsList = summary.map { ticker, val in
            let avgPrice = val.shares > 0 ? Int(val.invested / val.shares) : 0
            let lotsCount = Int(val.shares / 100)
            let nameDesc = val.name.isEmpty ? "" : " (\(val.name))"
            return "\(ticker)\(nameDesc): \(lotsCount) lot(s) / \(Int(val.shares)) shares @ avg buy Rp \(avgPrice)"
        }.joined(separator: ", ")

        return "\(query)\n\n[User's Current Holdings on this device: \(holdingsList)]"
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.DarkPurpleAppBackground
                    .ignoresSafeArea()
                    .contentShape(Rectangle())
                    .onTapGesture {
                        isInputFocused = false
                    }

                VStack(spacing: 0) {
                    if !isChatting {
                        Spacer(minLength: 0)

                        headerText(isChatting: false)
                            .padding(.horizontal, 24)
                            .padding(.bottom, 20)

                        suggestionPills
                            .padding(.bottom, 24)
                    } else {
                        messageList
                            .transition(
                                .asymmetric(
                                    insertion: .move(edge: .bottom)
                                        .combined(with: .scale(scale: 0.92, anchor: .bottomTrailing))
                                        .combined(with: .opacity),
                                    removal: .opacity
                                )
                            )
                    }

                    inputBar
                }
                .animation(.spring(response: 0.85, dampingFraction: 0.88), value: viewModel.messages.isEmpty)
            }
            .simultaneousGesture(
                TapGesture().onEnded {
                    isInputFocused = false
                }
            )
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if isChatting {
                        Button {
                            withAnimation(.spring(response: 0.85, dampingFraction: 0.88)) {
                                viewModel.resetSession()
                            }
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.counterclockwise")
                                Text("Reset")
                            }
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(Color.white.opacity(0.75))
                        }
                    }
                }
            }
            .preferredColorScheme(.dark)
        }
    }

    private var isChatting: Bool {
        !viewModel.messages.isEmpty
    }

    // MARK: - Header Text
    private func headerText(isChatting: Bool) -> some View {
        Text("What financial insights\ncan i give you today?")
            .font(.system(size: 28, weight: .semibold, design: .rounded))
            .multilineTextAlignment(isChatting ? .leading : .center)
            .foregroundStyle(Color.white.opacity(isChatting ? 0.7 : 0.95))
            .lineSpacing(isChatting ? 1 : 4)
            .scaleEffect(isChatting ? 0.8 : 1.0, anchor: isChatting ? .leading : .center)
    }

    // MARK: - Suggestion Pills
    private var suggestionPills: some View {
        VStack(alignment: .trailing, spacing: 10) {
            ForEach(sampleSuggestions, id: \.self) { suggestion in
                Button {
                    withAnimation(.spring(response: 0.85, dampingFraction: 0.88)) {
                        let enriched = enrichPromptIfNeeded(suggestion)
                        viewModel.send(suggestion, promptOverride: enriched)
                    }
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 12))
                            .foregroundStyle(Color.PrimaryYellow)
                        Text(suggestion)
                            .font(.system(size: 13.5, weight: .medium))
                            .foregroundStyle(Color.white.opacity(0.9))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(
                        Capsule()
                            .fill(Color.AICardBg)
                    )
                    .overlay(
                        Capsule()
                            .stroke(Color.white.opacity(0.12), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(.horizontal, 20)
    }

    // MARK: - Message List
    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 0) {
                        headerText(isChatting: true)
                        Spacer(minLength: 0)
                    }
                    .padding(.leading, 4)
                    .padding(.top, 8)
                    .padding(.bottom, 4)

                    ForEach(viewModel.messages) { msg in
                        ChatBubbleRow(message: msg)
                            .id(msg.id)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 4)
                .padding(.bottom, 16)
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: viewModel.messages.count) { _, _ in
                if let last = viewModel.messages.last {
                    withAnimation(.easeOut(duration: 0.3)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    // MARK: - Input Bar
    private var inputBar: some View {
        HStack(spacing: 10) {
            TextField(
                "",
                text: $viewModel.inputText,
                prompt: Text("Ask anything...").foregroundColor(Color.white.opacity(0.4))
            )
            .font(.system(size: 15))
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 11)
            .background(Color.AICardBg)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.white.opacity(0.15), lineWidth: 1)
            )
            .focused($isInputFocused)
            .onSubmit {
                guard !isSendDisabled else { return }
                withAnimation(.spring(response: 0.85, dampingFraction: 0.88)) {
                    let text = viewModel.inputText
                    let enriched = enrichPromptIfNeeded(text)
                    viewModel.send(text, promptOverride: enriched)
                }
            }

            Button {
                withAnimation(.spring(response: 0.85, dampingFraction: 0.88)) {
                    let text = viewModel.inputText
                    let enriched = enrichPromptIfNeeded(text)
                    viewModel.send(text, promptOverride: enriched)
                }
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(isSendDisabled ? Color.white.opacity(0.2) : Color.PrimaryYellow)
            }
            .disabled(isSendDisabled)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var isSendDisabled: Bool {
        viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty || viewModel.isProcessing
    }
}

#Preview {
    ChatbotView()
}
